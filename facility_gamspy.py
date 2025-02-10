import sys
import gamspy as gp
import os
import time
import logging

logging.disable(logging.WARNING)


def solve_facility(solver, G, F):
    M = 2 * 1.414
    m = gp.Container()
    with m:
        grid = gp.Set(records=range(G + 1))
        grid2 = gp.Alias(alias_with=grid)
        facs = gp.Set(records=range(1, F + 1))
        dims = gp.Set(records=range(1, 3))
        y = gp.Variable(domain=[facs, dims], name='y')
        y.lo = 0
        y.up = 1
        s = gp.Variable(domain=[grid, grid2, facs], name='s')
        s.lo = 0
        z = gp.Variable(domain=[grid, grid2, facs], type=gp.VariableType.BINARY, name='z')
        r = gp.Variable(domain=[grid, grid2, facs, dims], name='r')
        d = gp.Variable(name='d')

        assmt = gp.Equation(domain=[grid, grid2], name='assmt')
        assmt[...] = gp.Sum(facs, z[grid, grid2, facs]) == 1

        quadrhs = gp.Equation(domain=[grid, grid2, facs], name='quadrhs')
        quadrhs[...] = s[grid, grid2, facs] == d + M * (1 - z[grid, grid2, facs])

        quaddistk1 = gp.Equation(domain=[grid, grid2, facs], name='quaddistk1')
        quaddistk1[...] = r[grid, grid2, facs, "1"] == (1 * grid.val) / G - y[facs, "1"]

        quaddistk2 = gp.Equation(domain=[grid, grid2, facs], name='quaddistk2')
        quaddistk2[...] = (
            r[grid, grid2, facs, "2"] == (1 * grid2.val) / G - y[facs, "2"]
        )

        quaddist = gp.Equation(domain=[grid, grid2, facs], name='quaddist')
        quaddist[...] = (
            r[grid, grid2, facs, "1"] ** 2 + r[grid, grid2, facs, "2"] ** 2
                <= s[grid, grid2, facs] ** 2
        )

        model = gp.Model(
            name="facility",
            equations=m.getEquations(),
            problem=gp.Problem.MIQCP,
            sense=gp.Sense.MIN,
            objective=d,
        )

        # d.lo = 1.4

        model.solve(solver=solver, solver_options={'writemps': 'gamspy.mps'}, options=gp.Options(listing_file='gamspy.lst', equation_listing_limit=100))


def main(Ns=[2]):
    solver = sys.argv[1]
    if solver not in ("gurobi", "copt"):
        raise ValueError(f"Unknown solver {solver}.")

    dir = os.path.realpath(os.path.dirname(__file__))
    for n in Ns:
        start = time.time()
        _ = solve_facility(solver, n, n)
        run_time = round(time.time() - start, 1)
        content = f"gamspy fac-{n} -1 {run_time}"
        print(content)
        with open(dir + "/benchmarks.csv", "a") as io:
            io.write(f"{content}\n")
    return


main()
