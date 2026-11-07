// Conversion to the organisers' "EVALUATION RESULTS — JSON FORMAT 1.0".
// Internal criterion IDs of the official checklist differ from the codes in the
// submission format, and internal evidence indices are 0-based while
// evidence_turns are 1-based over the original transcript array.
import { scoreResults, finalResults } from './core.mjs';
import { OFFICIAL_CHECKLIST_ID } from './official-checklist.mjs';

export const CRITERION_MAP = Object.freeze({
  contact: 'contact',
  needs: 'discovery',
  identification: 'identification',
  consultation: 'consultation',
  empathy: 'empathy',
  management: 'dialogue',
  closing: 'closure',
  gross_errors: 'critical'
});
export const SUBMISSION_ORDER = ['contact','discovery','identification','consultation','empathy','dialogue','closure','critical'];
const ALLOWED = {
  contact: v => typeof v === 'boolean',
  discovery: v => v === null || typeof v === 'boolean',
  identification: v => v === null || typeof v === 'boolean',
  consultation: v => v === null || [0,5,10,15,20].includes(v),
  empathy: v => typeof v === 'boolean',
  dialogue: v => [0,4,7,10].includes(v),
  closure: v => v === null || typeof v === 'boolean',
  critical: v => typeof v === 'boolean'
};
const MODES = ['local','external_api','hybrid'];

/** The records array of an organiser payload, without validating individual records. */
export function rawRecords(payload) {
  const records = Array.isArray(payload) ? payload : payload?.data?.records || payload?.records || (payload?.conversation_id ? [payload] : null);
  if (!Array.isArray(records)) throw new Error('Expected a records array (data.records, records or a top-level array).');
  return records;
}
/** Conversation IDs exactly as the API returned them, kept even if a record is malformed. */
export function datasetIds(payload) {
  return rawRecords(payload).map(r => r?.conversation_id).filter(id => typeof id === 'string' && id.trim());
}

function evidenceTurns(result, record) {
  const turns = new Set([...(result.evidence || []), ...(result.contextEvidence || [])]);
  // A missing action has no evidence: point at the inspected operator/system turns instead.
  if (!turns.size && result.checkedRange) {
    for (let i = result.checkedRange.start; i <= result.checkedRange.end; i++) {
      if (['operator','system'].includes(record?.transcript?.[i]?.speaker)) turns.add(i);
    }
    if (!turns.size) turns.add(result.checkedRange.start);
  }
  return [...turns].filter(Number.isInteger).sort((a,b) => a-b).map(i => i + 1);
}

/** Convert one internal evaluation into a submission entry (status ok or error). */
export function toSubmissionEntry(evaluation, { state = null, useAppeals = false } = {}) {
  const id = evaluation.conversationId;
  const checklist = evaluation.checklist;
  const results = useAppeals && state ? finalResults(state, evaluation) : evaluation.results;
  const missing = Object.keys(CRITERION_MAP).filter(k => !checklist.criteria.some(c => c.id === k));
  if (missing.length) return errorEntry(id, `Checklist ${checklist.id} lacks official criteria: ${missing.join(', ')}.`);
  const pending = results.filter(r => CRITERION_MAP[r.criterionId] && r.answer === 'pending');
  if (pending.length) return errorEntry(id, `Model returned "pending" (insufficient evidence) for: ${pending.map(r => CRITERION_MAP[r.criterionId]).join(', ')}.`);
  const official = { ...checklist, criteria: checklist.criteria.filter(c => CRITERION_MAP[c.id]) };
  const score = scoreResults(official, results.filter(r => CRITERION_MAP[r.criterionId]));
  const byCode = new Map(results.filter(r => CRITERION_MAP[r.criterionId]).map(r => [CRITERION_MAP[r.criterionId], r]));
  const criteria = SUBMISSION_ORDER.map(code => {
    const r = byCode.get(code);
    return { criterion_id: code, value: r.answer, rationale: r.explanation, evidence_turns: evidenceTurns(r, evaluation.record) };
  });
  const critical = byCode.get('critical').answer === true;
  return {
    conversation_id: id,
    status: 'ok',
    criteria,
    score: { final_score: score.total === null ? null : Number(score.total.toFixed(2)), critical_flag: critical }
  };
}
export function errorEntry(id, error) {
  return { conversation_id: id, status: 'error', criteria: [], score: null, error: String(error || 'Unknown processing error.') };
}

/**
 * Build the whole submission file.
 * ids: the conversation IDs of the dataset, in API order. Every ID gets exactly one entry.
 */
export function buildSubmission({ state, ids = null, teamName, modelName, mode = 'local', engine = 'model', checklistId = OFFICIAL_CHECKLIST_ID, useAppeals = false }) {
  if (!teamName) throw new Error('team_name is required.');
  if (!modelName) throw new Error('model name is required.');
  if (!MODES.includes(mode)) throw new Error(`mode must be one of ${MODES.join(', ')}.`);
  const evaluations = state.evaluations || [];
  const errors = state.errors || [];
  const wanted = ids || [...new Set([...(state.records || []).map(r => r.conversation_id), ...errors.map(e => e.conversationId)])];
  // Internal records store trimmed IDs; the submission keeps the API's exact ID string.
  const same = (a, b) => typeof a === 'string' && typeof b === 'string' && a.trim() === b.trim();
  const results = [...new Set(wanted)].map(id => {
    const current = (state.records || []).find(r => same(r.conversation_id, id));
    const e = evaluations.filter(e => same(e.conversationId, id) && e.engine === engine && e.checklist?.id === checklistId
      && (!current || JSON.stringify(current) === JSON.stringify(e.record))).at(-1);
    if (e) {
      try { const entry = toSubmissionEntry(e, { state, useAppeals }); entry.conversation_id = id; return entry; }
      catch (error) { return errorEntry(id, `Invalid evaluation: ${error.message}`); }
    }
    const failure = errors.filter(x => same(x.conversationId, id)).at(-1);
    return errorEntry(id, failure ? failure.error : 'Conversation was not evaluated.');
  });
  return { schema_version: '1.0', team_name: teamName, model: { mode, name: modelName }, results };
}

/** Returns a list of problems; empty means the file follows format 1.0. */
export function validateSubmission(file, { expectIds = null, expectCount = null } = {}) {
  const problems = [];
  if (!file || typeof file !== 'object') return ['Root must be an object.'];
  if (file.schema_version !== '1.0') problems.push('schema_version must be "1.0".');
  if (typeof file.team_name !== 'string' || !file.team_name.trim()) problems.push('team_name is required.');
  if (!file.model || !MODES.includes(file.model.mode) || typeof file.model.name !== 'string' || !file.model.name.trim()) problems.push('model.mode/model.name are invalid.');
  if (!Array.isArray(file.results)) return [...problems, 'results must be an array.'];
  const seen = new Set();
  for (const [i, r] of file.results.entries()) {
    const at = `results[${i}] (${r?.conversation_id})`;
    if (typeof r?.conversation_id !== 'string' || !r.conversation_id) { problems.push(`${at}: conversation_id missing.`); continue; }
    if (seen.has(r.conversation_id)) problems.push(`${at}: duplicate conversation_id.`);
    seen.add(r.conversation_id);
    if (r.status === 'error') {
      if (!Array.isArray(r.criteria) || r.criteria.length || r.score !== null || typeof r.error !== 'string' || !r.error) problems.push(`${at}: error entries need criteria [], score null and an error message.`);
      continue;
    }
    if (r.status !== 'ok') { problems.push(`${at}: status must be "ok" or "error".`); continue; }
    if (!Array.isArray(r.criteria) || r.criteria.length !== 8) { problems.push(`${at}: exactly 8 criteria required.`); continue; }
    const codes = r.criteria.map(c => c.criterion_id);
    for (const code of SUBMISSION_ORDER) if (codes.filter(c => c === code).length !== 1) problems.push(`${at}: criterion ${code} must appear exactly once.`);
    for (const c of r.criteria) {
      if (!ALLOWED[c.criterion_id]) { problems.push(`${at}: unknown criterion ${c.criterion_id}.`); continue; }
      if (!Object.hasOwn(c, 'value') || !ALLOWED[c.criterion_id](c.value)) problems.push(`${at}: invalid value for ${c.criterion_id}: ${JSON.stringify(c.value)}.`);
      if (typeof c.rationale !== 'string' || !c.rationale.trim()) problems.push(`${at}: rationale missing for ${c.criterion_id}.`);
      if (!Array.isArray(c.evidence_turns) || c.evidence_turns.some(t => !Number.isInteger(t) || t < 1)) problems.push(`${at}: evidence_turns for ${c.criterion_id} must be 1-based integers.`);
    }
    const fs = r.score?.final_score;
    if (!r.score || !(fs === null || (typeof fs === 'number' && fs >= 0 && fs <= 100 && Number(fs.toFixed(2)) === fs))) problems.push(`${at}: final_score must be 0–100 with at most 2 decimals.`);
    const crit = r.criteria.find(c => c.criterion_id === 'critical');
    if (r.score && crit && r.score.critical_flag !== crit.value) problems.push(`${at}: critical_flag must equal the critical criterion value.`);
  }
  if (expectIds) {
    for (const id of expectIds) if (!seen.has(id)) problems.push(`Missing conversation_id ${id}.`);
    for (const id of seen) if (!expectIds.includes(id)) problems.push(`Unexpected conversation_id ${id}.`);
  }
  if (expectCount !== null && file.results.length !== expectCount) problems.push(`Expected exactly ${expectCount} entries, found ${file.results.length}.`);
  return problems;
}
