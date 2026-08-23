"""CSV recording of navigation convergence for the NID + CLF-CBF controllers.

One trial writes three files into `log_dir`:

  trial_<id>_samples.csv  one row per control tick while a navigation sequence runs
  trial_<id>_events.csv   one row per stage transition and per sequence completion
  trial_<id>_meta.json    the controller knobs the trial ran with

Rows are flushed as they are written, because the trial runner kills the
controller as soon as the sequence it cares about reports complete - a buffered
writer would lose the tail of every trial.

Two error columns, both in metres:
  err        distance from the look-ahead point (the end effector) to the FINAL
             target of the sequence - the stick pickup point, or the puck. This
             is the convergence signal; it is defined in every stage, so it is
             what the convergence plot is drawn from.
  err_stage  distance to whatever waypoint the current stage is actually driving
             at: the standoff point in stages 0-1, the final target in stages 2-3.

`sequence` and `stage` are recorded as written, so a trial can be sliced by
stage - the final approach (MOVE_TO_STICK stage 3) is the segment that shows the
NID law converging on the pickup point.
"""

import csv
import json
import os

SAMPLE_FIELDS = ['trial', 't', 'sequence', 'stage', 'x', 'y', 'theta',
                 'p_xl', 'p_yl', 'target_x', 'target_y', 'err', 'err_stage', 'v', 'w']
EVENT_FIELDS = ['trial', 't', 'sequence', 'stage', 'event',
                'err', 'err_stage', 'x', 'y', 'target_x', 'target_y']


class NavRecorder:
    """Per-trial navigation telemetry writer. Disabled instances are no-ops."""

    def __init__(self, log_dir, trial_id, clock, meta=None, logger=None):
        self.trial_id = trial_id
        self.clock = clock
        self.logger = logger
        self._t0 = None
        self._last_key = None  # (sequence, stage), for stage-transition events

        os.makedirs(log_dir, exist_ok=True)
        prefix = os.path.join(log_dir, f'trial_{trial_id}')
        self._sample_file = open(f'{prefix}_samples.csv', 'w', newline='')
        self._event_file = open(f'{prefix}_events.csv', 'w', newline='')
        self._sample_writer = csv.DictWriter(self._sample_file, fieldnames=SAMPLE_FIELDS)
        self._event_writer = csv.DictWriter(self._event_file, fieldnames=EVENT_FIELDS)
        self._sample_writer.writeheader()
        self._event_writer.writeheader()
        self._sample_file.flush()
        self._event_file.flush()

        with open(f'{prefix}_meta.json', 'w') as f:
            json.dump(dict(meta or {}, trial=trial_id), f, indent=2, sort_keys=True)

        if self.logger:
            self.logger.info(f"[record] Trial {trial_id}: writing navigation telemetry to {prefix}_*.csv")

    def _now(self):
        """Seconds since the first recorded row."""
        t = self.clock.now()
        if self._t0 is None:
            self._t0 = t
        return (t - self._t0).nanoseconds / 1e9

    def sample(self, sequence, stage, x, y, theta, p_xl, p_yl,
               target_x, target_y, err, err_stage, v, w):
        t = self._now()
        key = (sequence, stage)
        if key != self._last_key:
            # Stage boundaries are inferred here rather than hooked at each
            # transition site, so the controller only needs the one call.
            self._write_event(t, sequence, stage, 'stage_enter', err, err_stage, x, y, target_x, target_y)
            self._last_key = key
        self._sample_writer.writerow({
            'trial': self.trial_id, 't': f'{t:.4f}', 'sequence': sequence, 'stage': stage,
            'x': f'{x:.5f}', 'y': f'{y:.5f}', 'theta': f'{theta:.5f}',
            'p_xl': f'{p_xl:.5f}', 'p_yl': f'{p_yl:.5f}',
            'target_x': f'{target_x:.5f}', 'target_y': f'{target_y:.5f}',
            'err': f'{err:.5f}', 'err_stage': f'{err_stage:.5f}',
            'v': f'{v:.5f}', 'w': f'{w:.5f}'})
        self._sample_file.flush()

    def mark(self, sequence, stage, event, err, err_stage=None,
             x=None, y=None, target_x=None, target_y=None):
        """Record a named event - notably 'complete', whose `err` is the tolerance
        the sequence was actually marked complete at (the Table 1 figure)."""
        self._write_event(self._now(), sequence, stage, event, err, err_stage, x, y, target_x, target_y)
        if self.logger:
            self.logger.info(f"[record] {sequence} {event} at err={err:.4f} m")

    def _write_event(self, t, sequence, stage, event, err, err_stage, x, y, target_x, target_y):
        def fmt(val):
            return '' if val is None else f'{val:.5f}'
        self._event_writer.writerow({
            'trial': self.trial_id, 't': f'{t:.4f}', 'sequence': sequence, 'stage': stage,
            'event': event, 'err': fmt(err), 'err_stage': fmt(err_stage),
            'x': fmt(x), 'y': fmt(y), 'target_x': fmt(target_x), 'target_y': fmt(target_y)})
        self._event_file.flush()

    def close(self):
        for f in (self._sample_file, self._event_file):
            try:
                f.close()
            except Exception:
                pass


class NullRecorder:
    """Stand-in used when --record is off, so call sites need no guards."""

    def sample(self, *args, **kwargs):
        pass

    def mark(self, *args, **kwargs):
        pass

    def close(self):
        pass
