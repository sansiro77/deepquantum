import torch

import deepquantum as dq


def test_tdm_output_wires():
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.delay(0, ntau=2, inputs=[0.3, 0.1], convention='mzi')
    circuit.delay(1, ntau=3, inputs=[0.4, -0.2], convention='mzi')

    output_wires = circuit.tdm_output_wires(3)

    assert torch.equal(output_wires, torch.tensor([[2, 6], [7, 8], [9, 10]]))


def test_tdm_substate_matches_global_circuit():
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.s(0, r=0.2)
    circuit.bs([0, 1], [0.3, 0.1])
    circuit.delay(0, ntau=2, inputs=[0.4, 0.2], convention='mzi')
    circuit.delay(1, ntau=3, inputs=[0.5, -0.1], convention='mzi')
    circuit.loss_db(0, 0.2)
    circuit.to(torch.double)
    nstep = 5
    time_steps = torch.tensor([1, 3, 4])
    wires = torch.tensor([1, 0])

    actual_cov, actual_mean = circuit.tdm_substate(nstep, time_steps, wires)
    global_circuit = circuit.global_circuit(nstep)
    global_circuit.to(torch.double)
    expected_cov, expected_mean = global_circuit()
    output_wires = circuit.tdm_output_wires(nstep)[time_steps][:, wires].reshape(-1)
    indices = torch.cat([output_wires, output_wires + global_circuit.nmode])

    torch.testing.assert_close(actual_cov, expected_cov[:, indices[:, None], indices])
    torch.testing.assert_close(actual_mean, expected_mean[:, indices])
