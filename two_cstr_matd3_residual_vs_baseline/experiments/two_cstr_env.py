"""A lightweight two-CSTR series-with-recycle Gymnasium environment.

This experiment environment mirrors the equations in ``pcgym.model_classes``
but uses a local RK4 integrator, so the PPO experiments do not require
CasADi/JAX.  The original baseline class keeps flows fixed; the staged class
below exposes fresh-feed, recycle, and both cooling temperatures as actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:  # Keep the single-agent environment importable without the optional multi-agent dependency.
    from pettingzoo import ParallelEnv

    _PETTINGZOO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal single-agent installs.
    ParallelEnv = object  # type: ignore[assignment,misc]
    _PETTINGZOO_AVAILABLE = False


@dataclass
class TwoCSTRParameters:
    # Values inherited from pcgym.model_classes.cstr_series_recycle.
    C_O: float = 97.35
    T_O: float = 298.0
    V1: float = 1.0e-3
    V2: float = 2.0e-3
    U1A1: float = 0.461
    U2A2: float = 0.732
    rho: float = 1.05e3
    cp: float = 3.766
    k: float = 3.118e5
    E: float = 46.14
    # Conventional exothermic sign.  The original class stores +58.41 while
    # multiplying by -deltaH; using -58.41 makes the physical sign explicit.
    deltaH: float = -58.41
    R: float = 8.3145e-3

    # Fixed process flows for the first single-agent experiment.
    F: float = 2.0e-4
    L: float = 1.0e-4


class TwoCSTRFixedTargetEnv(gym.Env):
    """Centralized two-CSTR tracking task with a fixed product target."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: TwoCSTRParameters | None = None,
        target: tuple[float, float] = (92.0, 306.0),
        horizon: int = 60,
        dt: float = 1.0,
        action_low: float = 285.0,
        action_high: float = 325.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.params = params or TwoCSTRParameters()
        self.target = np.asarray(target, dtype=np.float64)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.initial_state = np.array([self.params.C_O, self.params.T_O, self.params.C_O, self.params.T_O])

        # [C1, T1, C2, T2, C2_sp, T2_sp, previous_Tc1, previous_Tc2]
        self.observation_space = spaces.Box(
            low=-np.ones(8, dtype=np.float32), high=np.ones(8, dtype=np.float32), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32))
        self._state = self.initial_state.copy()
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._step = 0
        if seed is not None:
            self.reset(seed=seed)

    @staticmethod
    def _scale_to_unit(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return 2.0 * (values - low) / (high - low) - 1.0

    def _observation(self) -> np.ndarray:
        state_low = np.array([0.0, 280.0, 0.0, 280.0])
        state_high = np.array([110.0, 340.0, 110.0, 340.0])
        target_low = np.array([80.0, 290.0])
        target_high = np.array([100.0, 320.0])
        state_obs = self._scale_to_unit(self._state, state_low, state_high)
        target_obs = self._scale_to_unit(self.target, target_low, target_high)
        return np.concatenate([state_obs, target_obs, self._prev_action]).astype(np.float32)

    def _action_to_physical(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        return self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)

    def _rhs(self, state: np.ndarray, full_action: np.ndarray) -> np.ndarray:
        p = self.params
        C1, T1, C2, T2 = state
        F, L, Tc1, Tc2 = full_action
        r1 = p.k * C1 * np.exp(-p.E / (p.R * T1))
        r2 = p.k * C2 * np.exp(-p.E / (p.R * T2))
        return np.array(
            [
                p.C_O * F / p.V1 + L * C2 / p.V1 - (F + L) * C1 / p.V1 - r1,
                p.T_O * F / p.V1 + L * T2 / p.V1
                - p.U1A1 * (T1 - Tc1) / (p.V1 * p.rho * p.cp)
                - (F + L) * T1 / p.V1
                - p.deltaH * r1 / (p.rho * p.cp),
                (F + L) * (C1 - C2) / p.V2 - r2,
                (F + L) * (T1 - T2) / p.V2
                - p.U2A2 * (T2 - Tc2) / (p.V2 * p.rho * p.cp)
                - p.deltaH * r2 / (p.rho * p.cp),
            ],
            dtype=np.float64,
        )

    def _integrate(self, action_physical: np.ndarray) -> None:
        full_action = np.array([self.params.F, self.params.L, *action_physical], dtype=np.float64)
        h = self.dt
        x = self._state
        k1 = self._rhs(x, full_action)
        k2 = self._rhs(x + 0.5 * h * k1, full_action)
        k3 = self._rhs(x + 0.5 * h * k2, full_action)
        k4 = self._rhs(x + h * k3, full_action)
        self._state = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state = self.initial_state.copy()
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._step = 0
        return self._observation(), {"state": self._state.copy(), "target": self.target.copy()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        action_physical = self._action_to_physical(action)
        previous_action = self._prev_action.copy()
        self._integrate(action_physical)
        self._step += 1

        concentration_error = (self._state[2] - self.target[0]) / 10.0
        temperature_error = (self._state[3] - self.target[1]) / 10.0
        smoothness_error = (action - previous_action) / 2.0
        reward = -(
            concentration_error**2
            + temperature_error**2
            + 0.01 * float(np.sum(smoothness_error**2))
        )
        self._prev_action = action
        terminated = False
        truncated = self._step >= self.horizon
        info = {
            "state": self._state.copy(),
            "action_physical": action_physical.copy(),
            "target": self.target.copy(),
            "tracking_error": np.array([concentration_error, temperature_error]),
        }
        return self._observation(), float(reward), terminated, truncated, info


class TwoCSTRStageOffsetEnv(gym.Env):
    """Two-CSTR concentration tracking task with a stage-2 output offset.

    This is intentionally different from a transport-delay model.  All
    actions act immediately.  During the startup stage CSTR1 is active while
    CSTR2 has no valid concentration output.  At ``t == c2_start_offset`` the
    stage handoff initializes CSTR2 from CSTR1, so ``C2(t) == C1(t)`` exactly
    at that instant.  The CSTR2 concentration target is activated only from
    that handoff time.

    The action is ``[F, L, Tc1, Tc2]``.  ``F`` is fresh feed, ``L`` is the
    CSTR2-to-CSTR1 recycle flow, and the two ``Tc`` values are cooling
    temperatures.  Temperature remains a dynamic state but has no target.
    The concentration target is a three-segment schedule, and the initial
    CSTR1 concentration/temperature are randomly perturbed at reset.

    The realistic variant also includes a first-order cooling-valve response,
    slowly varying (AR(1)) fresh-feed disturbances, noisy sensor readings,
    and bounded actuator noise.  Consequently the policy sees a noisy
    observation of the physical Markov state (a practical POMDP); this is
    intentional for robustness testing.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: TwoCSTRParameters | None = None,
        target_schedule: tuple[tuple[float, float], ...] = (
            (96.0, 94.5),
            (94.0, 92.0),
            (95.0, 93.0),
        ),
        c2_start_offset: int = 10,
        recycle_start_offset: int = 20,
        horizon: int = 180,
        dt: float = 1.0,
        flow_low: tuple[float, float] = (1.0e-4, 0.0),
        flow_high: tuple[float, float] = (4.0e-4, 2.0e-4),
        tc_low: float = 285.0,
        tc_high: float = 325.0,
        concentration_scale: float = 5.0,
        c1_weight: float = 0.8,
        c2_weight: float = 1.0,
        action_smoothness_weight: float = 0.01,
        initial_concentration_noise: float = 0.8,
        initial_temperature_noise: float = 1.5,
        cooling_valve_tau_s: float = 5.0,
        feed_disturbance_rho: float = 0.98,
        feed_concentration_disturbance_std: float = 0.08,
        feed_temperature_disturbance_std: float = 0.12,
        feed_concentration_disturbance_bound: float = 0.8,
        feed_temperature_disturbance_bound: float = 1.2,
        sensor_concentration_std: float = 0.05,
        sensor_temperature_std: float = 0.12,
        flow_actuator_noise_fraction: float = 0.015,
        cooling_actuator_noise_std: float = 0.20,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.params = params or TwoCSTRParameters()
        self.target_schedule = np.asarray(target_schedule, dtype=np.float64)
        if self.target_schedule.shape != (3, 2):
            raise ValueError("target_schedule must contain three (C1_target, C2_target) pairs")
        if np.any(self.target_schedule[:, 1] >= self.target_schedule[:, 0]):
            raise ValueError("C2 target must remain strictly below C1 target")
        self.c2_start_offset = int(c2_start_offset)
        self.recycle_start_offset = int(recycle_start_offset)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.flow_low = np.asarray(flow_low, dtype=np.float64)
        self.flow_high = np.asarray(flow_high, dtype=np.float64)
        self.tc_low = float(tc_low)
        self.tc_high = float(tc_high)
        self.concentration_scale = float(concentration_scale)
        self.c1_weight = float(c1_weight)
        self.c2_weight = float(c2_weight)
        self.action_smoothness_weight = float(action_smoothness_weight)
        self.initial_concentration_noise = float(initial_concentration_noise)
        self.initial_temperature_noise = float(initial_temperature_noise)
        self.cooling_valve_tau_s = float(cooling_valve_tau_s)
        self.feed_disturbance_rho = float(feed_disturbance_rho)
        self.feed_concentration_disturbance_std = float(feed_concentration_disturbance_std)
        self.feed_temperature_disturbance_std = float(feed_temperature_disturbance_std)
        self.feed_concentration_disturbance_bound = float(feed_concentration_disturbance_bound)
        self.feed_temperature_disturbance_bound = float(feed_temperature_disturbance_bound)
        self.sensor_concentration_std = float(sensor_concentration_std)
        self.sensor_temperature_std = float(sensor_temperature_std)
        self.flow_actuator_noise_fraction = float(flow_actuator_noise_fraction)
        self.cooling_actuator_noise_std = float(cooling_actuator_noise_std)
        if self.cooling_valve_tau_s <= 0.0 or not 0.0 <= self.feed_disturbance_rho < 1.0:
            raise ValueError("cooling_valve_tau_s must be positive and feed_disturbance_rho in [0, 1)")
        if not 0 <= self.c2_start_offset < self.recycle_start_offset <= self.horizon:
            raise ValueError("offsets must satisfy 0 <= c2_start_offset < recycle_start_offset <= horizon")

        self.initial_state = np.array(
            [self.params.C_O, self.params.T_O, self.params.C_O, self.params.T_O],
            dtype=np.float64,
        )
        # [C1, T1, C2_placeholder, T2_placeholder, current C1*, current C2*, previous F,
        #  previous L, previous Tc1, previous Tc2, CSTR2-active, recycle-active,
        #  normalized progress, measured feed concentration, measured feed temperature]
        self.observation_space = spaces.Box(
            low=-np.ones(15, dtype=np.float32),
            high=np.ones(15, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self._state = self.initial_state.copy()
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._step = 0
        self._handoff_value: float | None = None
        self._tc_actual = np.full(2, self.params.T_O, dtype=np.float64)
        self._feed_disturbance = np.zeros(2, dtype=np.float64)
        # Last applied physical actuator values.  The original observation
        # does not expose this field; the feedback-enhanced parallel wrapper
        # uses it to provide actuator-state feedback without changing the
        # original baseline environment interface.
        self._last_action_physical = np.array(
            [self.params.F, 0.0, self.params.T_O, self.params.T_O],
            dtype=np.float64,
        )

    @staticmethod
    def _scale_to_unit(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return 2.0 * (values - low) / (high - low) - 1.0

    def _observation(self) -> np.ndarray:
        state_low = np.array([0.0, 280.0, 0.0, 280.0], dtype=np.float64)
        state_high = np.array([110.0, 340.0, 110.0, 340.0], dtype=np.float64)
        target_low = np.array([80.0, 80.0], dtype=np.float64)
        target_high = np.array([100.0, 100.0], dtype=np.float64)
        state_for_obs = self._state.copy()
        if self._step < self.c2_start_offset:
            # CSTR2 has no concentration/temperature output before the stage
            # handoff.  Keep bounded placeholders and expose an explicit mask.
            state_for_obs[2:] = np.array([55.0, 310.0])
        else:
            state_for_obs[0] += self.np_random.normal(0.0, self.sensor_concentration_std)
            state_for_obs[1] += self.np_random.normal(0.0, self.sensor_temperature_std)
            state_for_obs[2] += self.np_random.normal(0.0, self.sensor_concentration_std)
            state_for_obs[3] += self.np_random.normal(0.0, self.sensor_temperature_std)
        if self._step < self.c2_start_offset:
            state_for_obs[0] += self.np_random.normal(0.0, self.sensor_concentration_std)
            state_for_obs[1] += self.np_random.normal(0.0, self.sensor_temperature_std)
        state_for_obs[0] = np.clip(state_for_obs[0], state_low[0], state_high[0])
        state_for_obs[1] = np.clip(state_for_obs[1], state_low[1], state_high[1])
        state_for_obs[2] = np.clip(state_for_obs[2], state_low[2], state_high[2])
        state_for_obs[3] = np.clip(state_for_obs[3], state_low[3], state_high[3])
        state_obs = self._scale_to_unit(state_for_obs, state_low, state_high)
        current_target = self._current_target()
        target_obs = self._scale_to_unit(current_target, target_low, target_high)
        stage2_active = 1.0 if self._step >= self.c2_start_offset else -1.0
        recycle_active = 1.0 if self._step >= self.recycle_start_offset else -1.0
        progress = 2.0 * self._step / self.horizon - 1.0
        feed_c = self.params.C_O + self._feed_disturbance[0] + self.np_random.normal(0.0, self.sensor_concentration_std)
        feed_t = self.params.T_O + self._feed_disturbance[1] + self.np_random.normal(0.0, self.sensor_temperature_std)
        feed_obs = self._scale_to_unit(
            np.array([feed_c, feed_t]),
            np.array([90.0, 290.0]),
            np.array([105.0, 310.0]),
        )
        return np.concatenate(
            [state_obs, target_obs, self._prev_action, np.array([stage2_active, recycle_active, progress]), feed_obs]
        ).astype(np.float32)

    def _current_target(self) -> np.ndarray:
        segment = min(self._step // 60, 2)
        return self.target_schedule[segment]

    def _action_to_physical(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(4)
        action = np.clip(action, -1.0, 1.0)
        flows = self.flow_low + 0.5 * (action[:2] + 1.0) * (self.flow_high - self.flow_low)
        cooling = self.tc_low + 0.5 * (action[2:] + 1.0) * (self.tc_high - self.tc_low)
        return np.concatenate([flows, cooling])

    def _rhs(
        self,
        state: np.ndarray,
        full_action: np.ndarray,
        c2_active: bool,
        recycle_active: bool,
        feed_concentration: float,
        feed_temperature: float,
    ) -> np.ndarray:
        # Same instantaneous equations as cstr_series_recycle.  Before the
        # handoff CSTR2 has no valid concentration.  Between the
        # handoff and the recycle-start time CSTR2 reacts, but its outlet has
        # not returned to CSTR1 yet.  Thus the effective recycle is zero until
        # t=20 s.
        p = self.params
        C1, T1, C2, T2 = state
        F, L, Tc1, Tc2 = full_action
        L_eff = L if recycle_active else 0.0
        r1 = p.k * C1 * np.exp(-p.E / (p.R * T1))
        r2 = p.k * C2 * np.exp(-p.E / (p.R * T2))
        if not c2_active:
            return np.array(
                [
                    feed_concentration * F / p.V1 - F * C1 / p.V1 - r1,
                    feed_temperature * F / p.V1
                    - p.U1A1 * (T1 - Tc1) / (p.V1 * p.rho * p.cp)
                    - F * T1 / p.V1
                    - p.deltaH * r1 / (p.rho * p.cp),
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                feed_concentration * F / p.V1 + L_eff * C2 / p.V1 - (F + L_eff) * C1 / p.V1 - r1,
                feed_temperature * F / p.V1 + L_eff * T2 / p.V1
                - p.U1A1 * (T1 - Tc1) / (p.V1 * p.rho * p.cp)
                - (F + L_eff) * T1 / p.V1
                - p.deltaH * r1 / (p.rho * p.cp),
                (F + L_eff) * (C1 - C2) / p.V2 - r2,
                (F + L_eff) * (T1 - T2) / p.V2
                - p.U2A2 * (T2 - Tc2) / (p.V2 * p.rho * p.cp)
                - p.deltaH * r2 / (p.rho * p.cp),
            ],
            dtype=np.float64,
        )

    def _integrate(
        self,
        action_physical: np.ndarray,
        c2_active: bool,
        recycle_active: bool,
        feed_concentration: float,
        feed_temperature: float,
    ) -> None:
        full_action = np.asarray(action_physical, dtype=np.float64)
        h = self.dt
        x = self._state
        k1 = self._rhs(x, full_action, c2_active, recycle_active, feed_concentration, feed_temperature)
        k2 = self._rhs(x + 0.5 * h * k1, full_action, c2_active, recycle_active, feed_concentration, feed_temperature)
        k3 = self._rhs(x + 0.5 * h * k2, full_action, c2_active, recycle_active, feed_concentration, feed_temperature)
        k4 = self._rhs(x + h * k3, full_action, c2_active, recycle_active, feed_concentration, feed_temperature)
        self._state = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state = self.initial_state.copy()
        self._state[0] += self.np_random.uniform(
            -self.initial_concentration_noise, self.initial_concentration_noise
        )
        self._state[1] += self.np_random.uniform(
            -self.initial_temperature_noise, self.initial_temperature_noise
        )
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._step = 0
        self._handoff_value = None
        self._tc_actual = np.full(2, self.params.T_O, dtype=np.float64)
        self._feed_disturbance = np.zeros(2, dtype=np.float64)
        self._last_action_physical = np.array(
            [self.params.F, 0.0, self.params.T_O, self.params.T_O],
            dtype=np.float64,
        )
        return self._observation(), {
            "state": self._state.copy(),
            "target": self._current_target().copy(),
            "c2_output_valid": False,
            "recycle_active": False,
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(4)
        action = np.clip(action, -1.0, 1.0)
        c2_active_before_step = self._step >= self.c2_start_offset
        recycle_active_before_step = self._step >= self.recycle_start_offset
        # L has no physical value before the CSTR2 outlet returns at t=20 s.
        # Keep the policy dimension present, but force the effective actuator
        # to zero until then.
        effective_action = action.copy()
        if not recycle_active_before_step:
            effective_action[1] = -1.0
        if not c2_active_before_step:
            effective_action[3] = -1.0
        previous_action = self._prev_action.copy()

        command_physical = self._action_to_physical(effective_action)
        flow_sigma = self.flow_actuator_noise_fraction * (self.flow_high - self.flow_low)
        F_actual = np.clip(
            command_physical[0] + self.np_random.normal(0.0, flow_sigma[0]),
            self.flow_low[0], self.flow_high[0]
        )
        L_actual = np.clip(
            command_physical[1] + self.np_random.normal(0.0, flow_sigma[1]),
            self.flow_low[1], self.flow_high[1]
        ) if recycle_active_before_step else 0.0
        alpha = min(self.dt / self.cooling_valve_tau_s, 1.0)
        self._tc_actual += alpha * (command_physical[2:] - self._tc_actual)
        self._tc_actual += self.np_random.normal(0.0, self.cooling_actuator_noise_std, size=2)
        self._tc_actual = np.clip(self._tc_actual, self.tc_low, self.tc_high)
        action_physical = np.array([F_actual, L_actual, *self._tc_actual], dtype=np.float64)
        self._last_action_physical = action_physical.copy()
        reported_action_physical = action_physical.astype(object)
        reported_command_physical = command_physical.astype(object)
        if not c2_active_before_step:
            reported_action_physical[3] = None
            reported_command_physical[3] = None
        if not recycle_active_before_step:
            reported_action_physical[1] = None
            reported_command_physical[1] = None

        feed_concentration = self.params.C_O + self._feed_disturbance[0]
        feed_temperature = self.params.T_O + self._feed_disturbance[1]

        # All four controls act immediately.  Before t=10 s only CSTR1 is
        # active; between t=10 and t=20 s CSTR2 reacts but its recycle has
        # not yet returned to CSTR1.
        self._integrate(
            action_physical,
            c2_active_before_step,
            recycle_active_before_step,
            feed_concentration,
            feed_temperature,
        )
        self._step += 1

        handoff = False
        if self._step == self.c2_start_offset:
            # The first valid CSTR2 concentration is exactly the CSTR1
            # concentration at the handoff time t=10 s.
            self._state[2] = self._state[0]
            self._state[3] = self._state[1]
            self._handoff_value = float(self._state[0])
            handoff = True

        current_target = self._current_target()
        c1_error = (self._state[0] - current_target[0]) / self.concentration_scale
        c2_error = (self._state[2] - current_target[1]) / self.concentration_scale
        smoothness_error = (effective_action - previous_action) / 2.0
        c2_active = self._step >= self.c2_start_offset
        reward = -(
            self.c1_weight * c1_error**2
            + (self.c2_weight * c2_error**2 if c2_active else 0.0)
            + self.action_smoothness_weight * float(np.sum(smoothness_error**2))
        )
        self._prev_action = effective_action

        self._feed_disturbance[0] = np.clip(
            self.feed_disturbance_rho * self._feed_disturbance[0]
            + self.feed_concentration_disturbance_std * self.np_random.normal(),
            -self.feed_concentration_disturbance_bound,
            self.feed_concentration_disturbance_bound,
        )
        self._feed_disturbance[1] = np.clip(
            self.feed_disturbance_rho * self._feed_disturbance[1]
            + self.feed_temperature_disturbance_std * self.np_random.normal(),
            -self.feed_temperature_disturbance_bound,
            self.feed_temperature_disturbance_bound,
        )

        F, L, _, _ = action_physical
        recycle_active = self._step >= self.recycle_start_offset
        if recycle_active:
            inlet_c1 = (F * feed_concentration + L * self._state[2]) / max(F + L, 1.0e-12)
        else:
            inlet_c1 = feed_concentration
        terminated = False
        truncated = self._step >= self.horizon
        info = {
            "state": self._state.copy(),
            "target": current_target.copy(),
            # action_physical is the applied numeric actuator signal.  The
            # companion available signal uses None for unavailable L/Tc2 so
            # plotting can leave those intervals blank without NaN-based
            # determinism issues in Gymnasium.
            "action_physical": action_physical.copy(),
            "action_physical_available": reported_action_physical,
            "action_physical_applied": action_physical.copy(),
            "action_physical_command": command_physical.copy(),
            "action_command_available": reported_command_physical,
            "action_normalized_raw": action.copy(),
            "action_normalized_effective": effective_action.copy(),
            "tracking_error": np.array([c1_error, c2_error]),
            "c2_output_valid": c2_active,
            "c2_handoff": handoff,
            "target_switch_event": bool(self._step in (60, 120)),
            "recycle_event": bool(self._step == self.recycle_start_offset),
            "c2_handoff_value": self._handoff_value,
            "c1_inlet_concentration": float(inlet_c1),
            "stage": 3 if recycle_active else (2 if c2_active else 1),
            "recycle_active": recycle_active,
            "effective_recycle_flow": float(L if recycle_active else 0.0),
            "feed_concentration": float(feed_concentration),
            "feed_temperature": float(feed_temperature),
            "feed_disturbance": self._feed_disturbance.copy(),
            # ``None`` explicitly represents "no CSTR2 concentration value"
            # before t=10 s and keeps Gymnasium's determinism check valid.
            "c2_concentration": float(self._state[2]) if c2_active else None,
        }
        return self._observation(), float(reward), terminated, truncated, info


class TwoCSTRParallelEnv(ParallelEnv):
    """PettingZoo parallel wrapper for distributed two-CSTR control.

    The physical model remains :class:`TwoCSTRStageOffsetEnv`; this class only
    changes the decision interface from one centralized action ``[F, L, Tc1,
    Tc2]`` to two simultaneous local actions:

    - ``cstr1``: ``[F, Tc1]``
    - ``cstr2``: ``[L, Tc2]``

    This is a cooperative CTDE interface.  Actors receive local observations,
    while ``info[agent]["global_state"]`` exposes a clean physical-state
    vector that can be used by a centralized MATD3 critic during training.
    It is intentionally not part of either actor's observation.

    The timing rules are inherited unchanged from the base environment:
    CSTR2 has no valid output for ``t < 10`` s and is initialized with
    ``C2(10) = C1(10)``; recycle flow ``L`` is unavailable for ``t < 20`` s.
    The corresponding action availability is reported in both ``info`` and
    the local observation flags.  The action dimensions remain fixed so that
    replay buffers stay synchronized across the episode.
    """

    metadata = {"name": "two_cstr_distributed_v1", "render_modes": []}
    possible_agents = ["cstr1", "cstr2"]

    def __init__(self, base_env: TwoCSTRStageOffsetEnv | None = None, **env_kwargs) -> None:
        if not _PETTINGZOO_AVAILABLE:
            raise ImportError(
                "TwoCSTRParallelEnv requires PettingZoo. Install it with "
                "`pip install -r requirements-experiments.txt`."
            )
        self.env = base_env or TwoCSTRStageOffsetEnv(**env_kwargs)
        self.agents = self.possible_agents[:]

        # Both actors use the same length and range to simplify multi-agent
        # replay-buffer batching.  Values are normalized to [-1, 1].
        # cstr1: own C1/T1, communicated C2/T2, targets, previous [F,Tc1],
        #        availability flags, progress, measured feed C/T.
        # cstr2: own C2/T2, communicated C1/T1, targets, previous [L,Tc2],
        #        the same flags/progress/feed measurements.
        local_low = -np.ones(13, dtype=np.float32)
        local_high = np.ones(13, dtype=np.float32)
        self.observation_spaces = {
            agent: spaces.Box(low=local_low, high=local_high, dtype=np.float32)
            for agent in self.possible_agents
        }
        # Optional PettingZoo global state interface.  MATD3 implementations
        # may use this directly for centralized critics instead of reading it
        # from the per-agent info dictionaries.
        self.state_space = spaces.Box(
            low=-np.ones(13, dtype=np.float32),
            high=np.ones(13, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_spaces = {
            agent: spaces.Box(
                low=-np.ones(2, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @staticmethod
    def _local_observations(base_observation: np.ndarray) -> dict[str, np.ndarray]:
        """Split the base noisy observation into two local actor inputs."""
        obs = np.asarray(base_observation, dtype=np.float32).reshape(15)
        # Base indices:
        # [C1,T1,C2,T2,C1*,C2*,prevF,prevL,prevTc1,prevTc2,
        #  c2_active,recycle_active,progress,feedC,feedT]
        # Reorder to keep the documented 13-vector layout:
        # own state (2), communicated state (2), targets (2), own previous
        # actions (2), stage/recycle flags (2), progress (1), feed (2).
        cstr1 = np.asarray(
            [
                obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[8],
                obs[10], obs[11], obs[12], obs[13], obs[14],
            ],
            dtype=np.float32,
        )
        cstr2 = np.asarray(
            [
                obs[2], obs[3], obs[0], obs[1], obs[4], obs[5], obs[7], obs[9],
                obs[10], obs[11], obs[12], obs[13], obs[14],
            ],
            dtype=np.float32,
        )
        return {"cstr1": cstr1, "cstr2": cstr2}

    def _global_state(self, base_info: dict) -> np.ndarray:
        """Return normalized physical state for a centralized critic."""
        state = np.asarray(base_info["state"], dtype=np.float64)
        state_norm = TwoCSTRStageOffsetEnv._scale_to_unit(
            state,
            np.array([0.0, 280.0, 0.0, 280.0], dtype=np.float64),
            np.array([110.0, 340.0, 110.0, 340.0], dtype=np.float64),
        )
        target = TwoCSTRStageOffsetEnv._scale_to_unit(
            np.asarray(base_info["target"], dtype=np.float64),
            np.array([80.0, 80.0], dtype=np.float64),
            np.array([100.0, 100.0], dtype=np.float64),
        )
        tc_actual = TwoCSTRStageOffsetEnv._scale_to_unit(
            np.asarray(self.env._tc_actual, dtype=np.float64),
            np.array([self.env.tc_low, self.env.tc_low], dtype=np.float64),
            np.array([self.env.tc_high, self.env.tc_high], dtype=np.float64),
        )
        feed_disturbance = np.array(
            [
                self.env._feed_disturbance[0] / self.env.feed_concentration_disturbance_bound,
                self.env._feed_disturbance[1] / self.env.feed_temperature_disturbance_bound,
            ],
            dtype=np.float64,
        )
        flags = np.array(
            [
                float(bool(base_info["c2_output_valid"])),
                float(bool(base_info["recycle_active"])),
                2.0 * self.env._step / self.env.horizon - 1.0,
            ],
            dtype=np.float64,
        )
        return np.concatenate([state_norm, target, tc_actual, feed_disturbance, flags]).astype(np.float32)

    def state(self) -> np.ndarray:
        """Return the current normalized physical state for CTDE critics."""
        base_info = {
            "state": self.env._state,
            "target": self.env._current_target(),
            "c2_output_valid": self.env._step >= self.env.c2_start_offset,
            "recycle_active": self.env._step >= self.env.recycle_start_offset,
        }
        return self._global_state(base_info)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        base_observation, base_info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        local_observations = self._local_observations(base_observation)
        global_state = self._global_state(base_info)
        action_mask = {
            "cstr1": np.array([1.0, 1.0], dtype=np.float32),
            "cstr2": np.array([0.0, 0.0], dtype=np.float32),
        }
        infos = {
            agent: {
                "global_state": global_state.copy(),
                "action_mask": action_mask[agent].copy(),
                "c2_output_valid": False,
                "recycle_active": False,
                "base_info": base_info,
            }
            for agent in self.possible_agents
        }
        return local_observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        if set(actions) != set(self.agents):
            missing = sorted(set(self.agents) - set(actions))
            extra = sorted(set(actions) - set(self.agents))
            raise ValueError(f"Expected actions for {self.agents}; missing={missing}, extra={extra}")
        a1 = np.clip(np.asarray(actions["cstr1"], dtype=np.float32).reshape(2), -1.0, 1.0)
        a2 = np.clip(np.asarray(actions["cstr2"], dtype=np.float32).reshape(2), -1.0, 1.0)
        # Joint order matches TwoCSTRStageOffsetEnv: [F, L, Tc1, Tc2].
        joint_action = np.asarray([a1[0], a2[0], a1[1], a2[1]], dtype=np.float32)
        base_observation, team_reward, terminated, truncated, base_info = self.env.step(joint_action)
        local_observations = self._local_observations(base_observation)
        global_state = self._global_state(base_info)
        c2_valid = bool(base_info["c2_output_valid"])
        recycle_active = bool(base_info["recycle_active"])
        action_mask = {
            "cstr1": np.array([1.0, 1.0], dtype=np.float32),
            "cstr2": np.array([float(recycle_active), float(c2_valid)], dtype=np.float32),
        }
        common = {
            "global_state": global_state,
            "joint_action_normalized": joint_action.copy(),
            "joint_action_physical": np.asarray(base_info["action_physical_applied"], dtype=np.float64).copy(),
            "action_mask": action_mask,
            "c2_output_valid": c2_valid,
            "recycle_active": recycle_active,
            "base_info": base_info,
        }
        infos = {agent: {**common, "action_mask": action_mask[agent].copy()} for agent in self.possible_agents}
        rewards = {agent: float(team_reward) for agent in self.possible_agents}
        terminations = {agent: bool(terminated) for agent in self.possible_agents}
        truncations = {agent: bool(truncated) for agent in self.possible_agents}
        if terminated or truncated:
            self.agents = []
        return local_observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self.env.close()


class TwoCSTRFeedbackParallelEnv(TwoCSTRParallelEnv):
    """Distributed wrapper with actuator and neighbor-action feedback.

    The physical model and reward are unchanged.  Each local observation
    retains the original 13 features and appends four normalized feedback
    values:

    - ``cstr1``: actual ``[F, Tc1]`` plus previous neighbor command ``[L, Tc2]``;
    - ``cstr2``: actual ``[L, Tc2]`` plus previous neighbor command ``[F, Tc1]``.

    The actual actuator values expose the cooling-valve lag and actuator noise;
    the neighbor command is a two-value low-bandwidth coordination signal.
    No global physical state is exposed to either actor.
    """

    metadata = {"name": "two_cstr_distributed_feedback_v1", "render_modes": []}

    def __init__(self, base_env: TwoCSTRStageOffsetEnv | None = None, **env_kwargs) -> None:
        super().__init__(base_env=base_env, **env_kwargs)
        local_low = -np.ones(17, dtype=np.float32)
        local_high = np.ones(17, dtype=np.float32)
        self.observation_spaces = {
            agent: spaces.Box(low=local_low, high=local_high, dtype=np.float32)
            for agent in self.possible_agents
        }

    def _feedback_observations(self, base_observation: np.ndarray) -> dict[str, np.ndarray]:
        base_local = self._local_observations(base_observation)
        raw = np.asarray(base_observation, dtype=np.float64).reshape(15)
        actual = np.asarray(self.env._last_action_physical, dtype=np.float64)
        actual_norm = np.concatenate(
            [
                TwoCSTRStageOffsetEnv._scale_to_unit(
                    actual[:2], self.env.flow_low, self.env.flow_high
                ),
                TwoCSTRStageOffsetEnv._scale_to_unit(
                    actual[2:],
                    np.array([self.env.tc_low, self.env.tc_low], dtype=np.float64),
                    np.array([self.env.tc_high, self.env.tc_high], dtype=np.float64),
                ),
            ]
        )
        # Base observation indices 6:10 are previous normalized [F,L,Tc1,Tc2].
        previous_command = raw[6:10].astype(np.float32)
        return {
            "cstr1": np.concatenate(
                [base_local["cstr1"], actual_norm[[0, 2]], previous_command[[1, 3]]]
            ).astype(np.float32),
            "cstr2": np.concatenate(
                [base_local["cstr2"], actual_norm[[1, 3]], previous_command[[0, 2]]]
            ).astype(np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        base_observation, base_info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        local_observations = self._feedback_observations(base_observation)
        global_state = self._global_state(base_info)
        action_mask = {
            "cstr1": np.array([1.0, 1.0], dtype=np.float32),
            "cstr2": np.array([0.0, 0.0], dtype=np.float32),
        }
        infos = {
            agent: {
                "global_state": global_state.copy(),
                "action_mask": action_mask[agent].copy(),
                "c2_output_valid": False,
                "recycle_active": False,
                "base_info": base_info,
            }
            for agent in self.possible_agents
        }
        return local_observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        if set(actions) != set(self.agents):
            missing = sorted(set(self.agents) - set(actions))
            extra = sorted(set(actions) - set(self.agents))
            raise ValueError(f"Expected actions for {self.agents}; missing={missing}, extra={extra}")
        a1 = np.clip(np.asarray(actions["cstr1"], dtype=np.float32).reshape(2), -1.0, 1.0)
        a2 = np.clip(np.asarray(actions["cstr2"], dtype=np.float32).reshape(2), -1.0, 1.0)
        joint_action = np.asarray([a1[0], a2[0], a1[1], a2[1]], dtype=np.float32)
        base_observation, team_reward, terminated, truncated, base_info = self.env.step(joint_action)
        local_observations = self._feedback_observations(base_observation)
        global_state = self._global_state(base_info)
        c2_valid = bool(base_info["c2_output_valid"])
        recycle_active = bool(base_info["recycle_active"])
        action_mask = {
            "cstr1": np.array([1.0, 1.0], dtype=np.float32),
            "cstr2": np.array([float(recycle_active), float(c2_valid)], dtype=np.float32),
        }
        common = {
            "global_state": global_state,
            "joint_action_normalized": joint_action.copy(),
            "joint_action_physical": np.asarray(base_info["action_physical_applied"], dtype=np.float64).copy(),
            "action_mask": action_mask,
            "c2_output_valid": c2_valid,
            "recycle_active": recycle_active,
            "base_info": base_info,
        }
        infos = {agent: {**common, "action_mask": action_mask[agent].copy()} for agent in self.possible_agents}
        rewards = {agent: float(team_reward) for agent in self.possible_agents}
        terminations = {agent: bool(terminated) for agent in self.possible_agents}
        truncations = {agent: bool(truncated) for agent in self.possible_agents}
        if terminated or truncated:
            self.agents = []
        return local_observations, rewards, terminations, truncations, infos


def parallel_env(**kwargs) -> TwoCSTRParallelEnv:
    """Convenience factory following the PettingZoo environment convention."""
    return TwoCSTRParallelEnv(**kwargs)
