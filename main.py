from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, BasicAer
import matplotlib.pyplot as plt


backend = BasicAer.get_backend('qasm_simulator')

q = QuantumRegister(2)
c = ClassicalRegister(2)

qc = QuantumCircuit(q, c)

qc.h(q[0])
qc.h(q[1])
qc.cz(q[0], q[1])
qc.h(q[0])
qc.h(q[1])
qc.x(q[0])
qc.x(q[1])
qc.cz(q[1], q[0])
qc.x(q[0])
qc.x(q[1])
qc.h(q[0])
qc.h(q[1])

qc.measure(q, c)

job = execute(qc, backend=backend, shots=1024)
result = job.result()

print(result.get_counts(qc))

qc.draw('mpl')
plt.show()
