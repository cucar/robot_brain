/**
 * Compatibility re-export - the real Job class now lives in the lib. Existing
 * job files that still import `./job.js` continue to work; new code should
 * import Job directly from '#brain-node'.
 */
export { Job } from '#brain-node';
