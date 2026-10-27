"""Inference-only action restriction for a same-checkpoint speed diagnostic.

The original physical state/features and loading/waiting masks are retained.
Only actual flight actions are restricted. A stranded rollout remains incomplete;
this is not a separately optimised single-speed routing algorithm.
"""

from revision_v3.environment import RouteEnvironment


class FixedSpeedEnvironment(RouteEnvironment):
    def __init__(self, *args, fixed_speed_mode, **kwargs):
        if type(fixed_speed_mode) is not int or fixed_speed_mode not in (0, 1):
            raise ValueError("Fixed speed mode must be integer 0 or 1")
        self.fixed_speed_mode = fixed_speed_mode
        super().__init__(*args, **kwargs)

    def raw_mask(self):
        # done(), get_mask() and step() all see this restriction, so an empty
        # legal set terminates without ever executing the categorical dummy.
        mask = super().raw_mask().clone()
        other = 1 - self.fixed_speed_mode
        mask[:, other:2*self.N:2] = False
        mask[:, self.return_offset + other] = False
        return mask


def assert_fixed_flights(actions, n_nodes, mode):
    """Check both delivery and depot-return flights in the exported trace."""
    for action in actions:
        if action < 2*n_nodes:
            actual = action % 2
        elif action in (3*n_nodes, 3*n_nodes+1):
            actual = action - 3*n_nodes
        else:
            continue
        if actual != mode:
            raise RuntimeError("Fixed-speed trace contains a forbidden flight mode")
