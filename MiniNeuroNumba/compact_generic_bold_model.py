# =======================================================================
# Compact Generic BOLD Simulator
#
# By Albert Juncà
# adapted by Gustavo Patow
# =======================================================================
import math
import numpy as np

from neuronumba.basic.attr import Attr
from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.monitors import RawSubSample, TemporalAverage
from neuronumba.simulator.simulator import Simulator
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerStochastic

from MiniNeuroNumba.compact_bold_simulator import CompactBoldSimulatorBase

# =======================================================================
# Compact_Simulator
# =======================================================================
class Compact_Simulator(CompactBoldSimulatorBase):

    g = Attr(required=True, doc="Coupling parameter")
    sigma = Attr(default=1e-03, doc="Noise amplitude")
    tr = Attr(required=True, doc="Actual TR in milliseconds")
    dt = Attr(default=0.1, doc="Delta time for the simulation in milliseconds")
    model = Attr(default=None, doc="If need to custom configure the model. It must be a Montbrio model")
    obs_var = Attr(default="r_e", doc="Observation variable")
    use_bold = Attr(default=True, doc="Perform a BOLD simulation, or directly return the activity")

    def _generate_bold(
        self,
        warmup_time: float,
        simulated_time: float
    ) -> np.ndarray:

        model = self.model
        model.configure(weights=self.weights, g=self.g)

        obs_var = self.obs_var
        n_roi = np.shape(self.weights)[0]

        # Prepare everything
        integrator = EulerStochastic(dt=self.dt, sigmas=model.get_noise_template() * self.sigma)
        con = Connectivity(
            weights=self.weights,
            lengths=np.random.rand(n_roi, n_roi)*10.0 + 1.0,
            speed=1.0
        )
        history = HistoryNoDelays()
        monitor = None
        if self.use_temporal_avg_monitor:
            monitor = TemporalAverage(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        else:
            monitor = RawSubSample(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        sim = Simulator(
            connectivity=con,
            model=model,
            history=history,
            integrator=integrator,
            monitors=[monitor]
        )

        # Run simulation
        sim.run(0, warmup_time + simulated_time)

        # Retreive simulated data and remove warmup
        sim_signal = monitor.data(obs_var)
        start_idx = int(sim_signal.shape[0] * warmup_time / (warmup_time + simulated_time))
        sim_signal = sim_signal[start_idx:, :]

        if not self.use_bold:
            return sim_signal

        # We can proceed to convert the signal to bold
        bold_converter = BoldStephan2008(tr=self.tr)
        bold_signal = bold_converter.compute_bold(sim_signal, monitor.period)

        return bold_signal
