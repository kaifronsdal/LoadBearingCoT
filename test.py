from sage.all import GF, PolynomialRing, Matrix, vector


def berlekamp_welch_RS(received, eval_points, message_degree, error_bound, F):
    N = len(eval_points)
    n = message_degree
    t = error_bound

    R = PolynomialRing(F, 'x')
    x = R.gen()

    # Q(x) has degree < n+t so it has n+t unknown coefficients
    # E(x) is monic of degree t so we have t unknown coefficients
    # In total, we have (n+t) + t = n+2*t unknowns
    num_unknowns = n + 2 * t

    # Build the linear system:
    # For each evaluation point a with corresponding received symbol r,
    # we require:
    #   Q(a) - rE(a) = 0
    # Write Q(x) = sum_{j=0}^{n+t-1} q_j x^j and
    #       E(x) = x^t + e_{t-1} x^(t-1) + ... + e_0
    # Then for each a we have
    #   (q_0 + q_1 a + ... + q_{n+t-1} a^(n+t-1))
    #     - r*(a^t + e_0 + e_1 a + ... + e_{t-1} a^(t-1)) = 0
    # Rearranging gives
    #   q_0 + q_1 a + ... + q_{n+t-1} a^(n+t-1)
    #     - r*e_0 - r*e_1 a - ... - r*e_{t-1} a^(t-1) = r*a^t

    M = Matrix(F, N, num_unknowns)
    bvec = vector(F, N)

    for i in range(N):
        a = F(eval_points[i])
        r = F(received[i])

        # The coefficients corresponding to Q(x)
        row = [a ** j for j in range(n + t)]

        # coefficients corresponding to rE(x)
        row += [-r * (a ** j) for j in range(t)]
        M[i] = vector(F, row)
        bvec[i] = r * (a ** t)

    # Solve the linear system M * sol = bvec.
    sol = M.solve_right(bvec)

    Q = sum(sol[j] * x ** j for j in range(n + t))
    Evec = sol[n + t: n + 2 * t]
    E = x ** t
    for j in range(t - 1, -1, -1):
        E += Evec[j] * x ** j

    # Q(x) = P(x)*E(x)
    P, remainder = Q.quo_rem(E)
    if remainder != 0:
        raise ValueError("Decoding error: nonzero remainder in division")

    decoded = [P(F(a)) for a in eval_points]
    return decoded, P, Q, E


def main():
    F = GF(113)
    eval_points = [F(i) for i in range(1, 17)]
    message_degree = 8
    error_bound = 4

    received_a = [52, 84, 35, 108, 70, 78, 43, 109, 66, 20, 100, 103, 11, 41, 14, 70]
    received_b = [7, 64, 58, 10, 90, 89, 99, 54, 42, 55, 82, 24, 35, 95, 38, 25]
    received_c = [81, 81, 93, 60, 27, 12, 37, 72, 68, 58, 67, 4, 76, 105, 49, 35]

    decoded_a, P_a, Q_a, E_a = berlekamp_welch_RS(received_a, eval_points, message_degree, error_bound, F)
    decoded_b, P_b, Q_b, E_b = berlekamp_welch_RS(received_b, eval_points, message_degree, error_bound, F)
    decoded_c, P_c, Q_c, E_c = berlekamp_welch_RS(received_c, eval_points, message_degree, error_bound, F)

    print("Decoded codeword (a):", [int(val) for val in decoded_a])
    print("Decoded codeword (b):", [int(val) for val in decoded_b])
    print("Decoded codeword (c):", [int(val) for val in decoded_c])


if __name__ == '__main__':
    main()
