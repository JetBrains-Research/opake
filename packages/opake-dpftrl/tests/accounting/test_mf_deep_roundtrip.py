"""MfGaussian's custom serializer fires under composition wrappers (#334)."""

from opake.api.accounting.core._accountant import Accountant
from opake.dpftrl.noise import identity_strategy
from opake.serialization import from_state_dict, state_dict


def test_repeated_mf_gaussian_accountant_round_trips():
    import opake.dpftrl.accounting as ftrl_acc

    mf = ftrl_acc.mf_gaussian(
        1.1, identity_strategy(), n_steps=8, min_sep=8, max_participations=1
    )
    acct = Accountant(prefix=mf * 8)
    sd = state_dict(acct)
    restored = from_state_dict(Accountant(), sd)
    assert restored.process == acct.process
