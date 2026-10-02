// Preload via `node --require`. Forces stdout/stderr into blocking writes so redirected
// output (`> file.log`) flushes as it's written instead of batching in Node's async pipe
// buffering on Windows, which can hide many minutes of real progress from a polled log file.
if (process.stdout._handle && process.stdout._handle.setBlocking) process.stdout._handle.setBlocking(true);
if (process.stderr._handle && process.stderr._handle.setBlocking) process.stderr._handle.setBlocking(true);
