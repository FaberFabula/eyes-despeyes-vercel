const FRAME_RATE = 60;
const LOW_FRAME_RATE = 30;

export function createFramePacer({ requestFrame = requestAnimationFrame, now = () => performance.now() } = {}) {
  let deadline = -Infinity;
  let started;
  let average = 0;
  let rate = FRAME_RATE;
  let streak = 0;

  function reset() {
    deadline = -Infinity;
    started = undefined;
    average = 0;
    rate = FRAME_RATE;
    streak = 0;
  }

  return {
    reset,
    next() {
      if (started !== undefined) {
        const work = Math.min(50, now() - started);
        average += 0.1 * (work - average);
        const canSwitch = rate === FRAME_RATE ? average > 14 : average < 9;
        streak = canSwitch ? streak + 1 : 0;
        if (streak >= (rate === FRAME_RATE ? 24 : 120)) {
          rate = rate === FRAME_RATE ? LOW_FRAME_RATE : FRAME_RATE;
          streak = 0;
        }
      }

      return new Promise((resolve) => {
        const interval = 1000 / rate;
        function deliver(timestamp) {
          // Allow minor display-clock jitter without dropping every other refresh.
          if (timestamp + 1 < deadline) {
            requestFrame(deliver);
            return;
          }
          deadline = timestamp - deadline >= interval ? timestamp + interval : deadline + interval;
          started = now();
          resolve();
        }
        requestFrame(deliver);
      });
    },
  };
}
