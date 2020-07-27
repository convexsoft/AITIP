import numpy as np
import parser
from util import *
from cadmm import Cadmm
import click
import sys, os, signal
from functools import reduce
import itertools
import string
from logger import Logger
import config

def get_proof(cadmm, l1, l2, l3, E, b, n, rvs):
    out = ''
    idx = np.bitwise_or(l1 > 1e-8, l1 < -1e-8)
    l_idx = np.where(idx)[0].tolist()
    l_val = l1[idx].tolist()

    out += 'The objective can be written as:\n'
    out += '\t{}\n'.format(parser.Combination.from_b(b, n, rvs, cadmm))
    for i, idx in enumerate(l_idx):
        Di = parser.Combination.from_b(cadmm.Arow(n, idx), n, rvs, cadmm)
        factor = l_val[i]
        if close(factor, -1):
            factor = '-'
        elif close(factor, 1):
            factor = ''
        out += '\t{}{}[{}]\n'.format('=' if i == 0 else '+', factor, Di)

    if E is not None:
        idx = np.bitwise_or(l3 > 1e-8, l3 < -1e-8)
        l3_idx = np.where(idx)[0].tolist()
        l3_val = l3[idx].tolist()

        for i, idx in enumerate(l3_idx):
            Di = parser.Combination.from_b(E[idx,:], n, rvs, cadmm)

            opt = '=' if len(l_idx) == 0 else '+'
            factor = -1 * l3_val[i]

            if opt == '+' and factor < 0:
                factor = abs(factor)
                opt = '-'

            if close(factor, 1):
                factor = ''

            out += '\t{}{}{{{}}}\n'.format(opt, factor, Di)

    out += '\t>=0\n'
    out += 'where expressions in [] are non-negative due to elemental inequalities,\n and expressions in {} are zero due to the user specified constraints.'
    return out

def get_disproof(cadmm, l1, l2, l3, E, b, n, rvs):
    out = ''
    m = get_m(n)
    idx = np.bitwise_or(l1 > 1e-8, l1 < -1e-8)
    l1_idx = np.where(idx)[0].tolist()
    l1_val = l1[idx].tolist()

    l2c = l2.copy()
    l2c[:m,:] = 0
    idx = np.bitwise_or(l2c > 1e-8, l2c < -1e-8)
    l2_idx = np.where(idx)[0].tolist()
    l2_val = l2c[idx].tolist()

    out += 'Canonical form: {} >= 0\n'.format(parser.Combination.from_b(b, n, rvs, cadmm))
    out += 'We cannot prove the given inequality, which means it is one of these two cases:\n'
    out += '\t1) The inequality is a non-Shannon-type inequality.\n'
    out += '\t2) The inequality is not true.\n'
    out += 'If it is case 2), a disproof can be constructed using the following hints:\n'
    out += 'A counter example can be constructed as a probability distribution satisfying all of the following conditions\n'
    for i, idx in enumerate(l1_idx):
        Di = parser.Combination.from_b(cadmm.Arow(n, idx), n, rvs, cadmm)
        out += '\t{} = 0\n'.format(Di)
    if E is not None:
        idx = np.bitwise_or(l3 > 1e-8, l3 < -1e-8)
        l3_idx = np.where(idx)[0].tolist()
        l3_val = l3[idx].tolist()
        for i, idx in enumerate(l3_idx):
            Di = parser.Combination.from_b(E[idx,:], n, rvs, cadmm)
            out += '\t{} = 0\n'.format(Di)
    for i, idx in enumerate(l2_idx):
        Di = parser.Combination.from_b(cadmm.Arow(n, idx), n, rvs, cadmm)
        out += '\t{} = 1\n'.format(Di)
    return out

def solve(in_str, in_lst, crossover=1, maxTime=1024, threads=0, gpu=False, odir=None, append=False, dev=False, shouldProve=True, gen=False, tol=1e-8):
    proved = 0
    logger = Logger(odir, 'a' if append else 'w+')

    cadmm = Cadmm(gpu)

    try:
        inequality = parser.parse(in_str)
    except ValueError as e:
        logger.log(e, 'err')
        raise e
    except:
        logger.log('We failed to parse your input. Please make sure you follow the input format.', 'err')
        raise ValueError('We failed to parse your input. Please make sure you follow the input format.')

    if type(inequality) is list:
        # input is one of the 3 macros
        if len(inequality) > 1:
            #input is macro 1 (Markov chain) and contains over 3 r.v.s., which is not allowed
            err = 'Markov chain is not allowed to be used as input inequality. You can only use Markov chain as a user-specified constraint.'
            logger.log(err, 'err')
            raise ValueError(err)
        else:
            inequality = inequality[0]

    if inequality.relation in ['=', 'eq']:
        ieq1 = inequality.copy()
        ieq2 = inequality.copy()
        ieq1.relation = '<='
        ieq2.relation = '>='
        
        out = 'Your input is an identity. We will try to prove your input by proving the following two inequalities:\n\t#1: {}\n\t#2: {}'.format(ieq1, ieq2)
        res1 = 0
        res2 = 0

        (res1,p1) = solve(str(ieq1), in_lst, crossover, maxTime, threads, gpu, odir, True, dev, shouldProve, gen, tol)

        if res1 == 0:
            out += '\nInequality #1 is not provable. A disproof can be found below.\n'
            out += p1
        elif res1 == -1:
            out += '\nError while trying to prove inequality #1.\n'
            out += p1
        else:
            (res2,p2) = solve(str(ieq2), in_lst, crossover, maxTime, threads, gpu, odir, True, dev, shouldProve, gen, tol)
            if res2 == 0:
                out += '\nInequality #2 is not provable. A disproof can be found below.\n'
                out += p2
            elif res2 == -1:
                out += '\Error while trying to prove inequality #2.\n'
                out += p2
            else:
                out += '\nThe input equality is true, as both inequalities #1 and #2 are provable. The proofs are shown below.\n'
                out += 'Inequality #1 Proof:\n'
                out += p1
                out += '\nInequality #2 Proof:\n'
                out += p2

        logger.log(out)

        new_res = 0
        if res1 == 1 and res2 == 1:
            new_res = 1
            logger.log(out, 'proof')
        elif res1 == -1 or res2 == -1:
            new_res = -1
            logger.log(out, 'err')
        else:
            new_res = 0
            logger.log(out, 'disproof')

        return (new_res, out)

    ps = []
    if len(in_lst) > 0:
        try:
            # ps = list(map(lambda x: parser.parse(x), in_lst))
            for instr in in_lst:
                parsed = parser.parse(instr)
                if type(parsed) == list:
                    ps += parsed
                else:
                    ps.append(parsed)
        except ValueError as e:
            logger.log(e, 'err')
        except:
            logger.log('We failed to parse your input. Please make sure you follow the input format.', 'err')
            raise ValueError('We failed to parse your input. Please make sure you follow the input format.')
        for p in ps:
            if p.relation not in ['=', 'eq']:
                logger.log('Your input {} is an inequality constraint. We support equality constraints only'.format(p), 'err')
                raise ValueError('Your input {} is an inequality constraint. We support equality constraints only'.format(p))


    user_rvs = reduce(lambda l,r: l + r.rvs(), ps, [])
    extra_rvs = list(set(user_rvs) - set(inequality.rvs()))
    inequality.add_rvs(extra_rvs)

    rvs = inequality.rvs()

    n = len(rvs)

    def get_common(n, rvs, ieq, usr_cons):
        ieqs = [ieq] + usr_cons
        cs = map(lambda i: i.canonical().push_left().lhs, ieqs)
        ms = list(reduce(lambda l,c: l + c.measures, cs, []))
        groups = []
        for etp in ms:
            groups.append(etp.variables)

        k = get_k(n)
        klst = np.zeros(k)

        for g in groups:
            idx = list(map(lambda v: 2 ** rvs.index(v), g))
            count = len(idx)
            coms = []
            for c in range(1, count + 1):
                coms += itertools.combinations(idx, c)
            com_idx = list(map(lambda c: reduce(lambda r,x: r | x, c, 0) - 1, coms))
            klst[com_idx] += 1

        common = cadmm.build_common_ary(n, klst)
        return common

    def pre_optimize(n, rvs, ieq, usr_cons):
        cs = get_common(n, rvs, ieq, usr_cons)
        common_idx = list(map(lambda c: cadmm.get_elements(c), cs))
        common_idx = [list(map(lambda x: int(math.log(x, 2)), l)) for l in common_idx]
        common_rvs = [list(map(lambda x: rvs[x], l)) for l in common_idx]
        common_rvs = list(filter(lambda l: len(l) > 1, common_rvs))
        logger.log('r.v.s that can be grouped together: {}'.format(common_rvs))

        candidates = sorted(set(string.ascii_letters) - set(rvs))
        replace = candidates[:len(common_rvs)]
        logger.log('replacing with: {}'.format(replace))

        ieqs = [ieq] + usr_cons
        ieqs = list(map(lambda x: x.canonical().push_left(), ieqs))

        for iq in ieqs:
            for i, crvs in enumerate(common_rvs):
                iq.replace_var(crvs, replace[i])

        return (ieqs[0], ieqs[1:], common_rvs, replace)

    logger.log('optimizing the input to reduce the problem size')

    common_rvs = []
    replacement = []
    (inequality, ps, common_rvs, replacement) = pre_optimize(n, rvs, inequality, ps)

    out = '\n'
    if len(common_rvs):
        out += 'The following set(s) of variables always appear together, so they have been replaced by new variable(s) to reduce the problem size\n'
        for i, crvs in enumerate(common_rvs):
            out += '\t{{{}}} ==> {}\n'.format(', '.join(crvs), replacement[i])
    out += 'Input: {}\n'.format(inequality)
    if len(ps):
        out += 'User constraints:\n'
        for p in ps:
            out += '\t{}\n'.format(p)

    rvs = inequality.rvs()
    b = inequality.b()

    np.savetxt("b.csv", b, delimiter=",")

    n = b2n(b)
    out_proof = None

    if config.max_n > 0 and n > config.max_n:
        max_n_out = 'Your input contains {} random variables. To save the resources on the server, you are only allowed to solve inequalities containing up to {} variables. To solve larger problem, please download our source code and run it on your own machine.'.format(n, config.max_n)
        logger.log(max_n_out, 'err')
        print(max_n_out)
        return (-1, max_n_out)

    if n == 1:
        obj = parser.Combination.from_b(b, n, rvs, cadmm)
        factor = 0
        if len(obj.measures) > 0:
            # it is necassary to check obj.measures is not empty. if the input is '0H(A) >= 0' it would be empty
            factor = obj.measures[0].factor
        out += 'The objective can be written as:\n'
        out += '\t{}\n'.format(obj)
        out += 'There is only 1 random variable, and the factor is {}, so the inequality is {}\n'.format('non-negative' if factor >= 0 else 'negative', 'true' if factor >= 0 else 'not true')
        logger.log(out)
        if factor >= 0:
            proved = 1
            logger.log(out, 'proof')
        else:
            logger.log(out, 'disproof')
        out_proof = out
    else:
        k = get_k(n)
        m = get_m(n)

        logger.log(n, 'n')

        E = None

        for p in ps:
            pb = p.b(n, rvs)
            if E is not None:
                E = np.vstack((E, pb))
            else:
                E = pb.copy()

        if E is not None:
            E = E.reshape((int(E.size/k), k))
            np.savetxt("E.csv", E, delimiter=",")

        if gen and dev:
            print(rvs)
            print('Exiting because the -generate option is on')
            print(out)
            sys.exit(0)

        (rc, obj, x, y, l1, l2, l3, output) = cadmm.solve(b, E, int(crossover), maxTime, tol, threads, dev)

        def density(v):
            return (np.where(np.abs(v) > 1e-6)[0].size)/v.size

        if E is None:
            print('l1, l2 density:', 0.5 * (density(l1) + density(l2)))
        else:
            print('l1, l2, l3 density:', 1/3 * (density(l1) + density(l2) + density(l3)))

        if not dev:
            logger.log(output)

        if rc == 1:
            time_out_out = 'Maxtimum solving time {} secs reached. No proof/disproof can be construted'.format(maxTime)
            logger.log(time_out_out, 'err')
            print(time_out_out)
            return (-1, time_out_out)
        if rc == 2:
            logger.log('Insufficient time for corssover. Raw ADMM results will be used to construct the proof/dirproof', 'err')
            print('Insufficient time for corssover. Raw ADMM results will be used to construct the proof/dirproof')

        proved = int(obj < 1e-7 and obj > -1e-7)

        if not (dev and not shouldProve):
            if obj < 1e-7 and obj > -1e-7:
                out += get_proof(cadmm, l1, l2, l3, E, b, n, rvs)
                logger.log(out, 'proof')
                logger.log(out)
            else:
                out += get_disproof(cadmm, l1, l2, l3, E, b, n, rvs)
                logger.log(out, 'disproof')
                logger.log(out)
            out_proof = out

    logger.close()
    return (proved, out_proof)


@click.command()
@click.option('-i', help='Objective inequality', type=str)
@click.option('-u', help='User constraints, separated with /', type=str)
@click.option('-n', help='Generate a random inequality invoving n r.v.s. Ignores -i and -u', type=int, default=0)
@click.option('--nocross', help='[Flag] Disable crossover', is_flag=True)
@click.option('-t', help='Maximum running time in seconds. Default: 1024', type=float, default=1024)
@click.option('--th', help='Number of threads to use. Set to 0 to use all threads. Default: 0', type=int, default=0)
@click.option('--gpu', help='[Flag] Enable GPU acceleration', is_flag=True)
@click.option('-o', help='Directory to store the output files', type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--debug', help='[Flag] Debug mode', is_flag=True)
@click.option('--noproof', help='[Flag] Skip proof/disproof generation. Only works when --debug is on', is_flag=True)
@click.option('--generate', help='[Flag] Generate E.csv and b.csv only without actually solving the problem. Only works when --debug is on', is_flag=True)
@click.option('--tol', help='ADMM tolerance. Default: 1e-8', type=float, default=1e-8)
def start(i, u, n, nocross, t, th, gpu, o, debug, noproof, generate, tol):
    if (not debug and (noproof or generate)):
        print('--noproof and --generate only work when --debug is on. Exiting...')
        return
    if n == 1:
        print('n must be at least 2')
        return
    if n > 26:
        print('n is too large')
        return
    if n > 1:
        cadmm = Cadmm(gpu)
        b = cadmm.random_b(n)
        i = str(parser.Combination.from_b(b, n, string.ascii_lowercase[:n], cadmm)) + '>=0'
        u = None

    print('GPU acceleration ' + ('enabled' if gpu else 'disabled'))
    solve(i, [] if u is None else u.split('/'), not nocross, t, th, gpu, o, False, debug, not noproof, generate, tol)

if __name__ == '__main__':
    sys.setrecursionlimit(30000)
    if len(sys.argv) == 1:
        in_str = input('Please input your inequality:\n')
        in_lst = minput('Please input your equality constraints (optional):\n')
        solve(in_str, in_lst)
    else:
        start()
