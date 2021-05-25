import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import abc
import random
import inspect
import itertools
import time
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed


#-------------------------------------------List of symbols------------------------------------------------
#Reputations
p_i = sym.Symbol('p_i')
p_ii = sym.Symbol('p_ii')
p_ik = sym.Symbol('p_ik')
p_j = sym.Symbol('p_j')
p_mr = sym.Symbol('p_mr')
p_rm = sym.Symbol('p_rm')
p_L = sym.Symbol("p_L")
p_noL = sym.Symbol("p_noL")

#Symbols---------------------------------
#Error rate
#Assessment error
eps_a = sym.Symbol("eps_a")
#Execution error
eps_e = sym.Symbol("eps_e")
#Proportion and number of observers
q = sym.Symbol("q")
N_o = sym.Symbol("N_o")
#benefit of coop
b = sym.Symbol("b")
#cost of coop
c = sym.Symbol("c")
#Generic symbol, in case it is needed
x = sym.Symbol('x')

#Probabilities matrix that describes the probability of each encounter: 11, 10, 01, 00 or 1,0
proba_matrix_second_order = sym.Matrix([[p_i * p_j, p_i * (1 - p_j), (1 - p_i) * p_j, (1 - p_i) * (1 - p_j)]])
proba_matrix_first_order = sym.Matrix([[p_j, (1 - p_j)]])

#------------------------------------------------------FUNCTIONS------------------------------------------------------------------------
#To extract parameters from the name of a file
def read_parameters(name_file):
    parameters= dict()
    #We ignore the first part which is the name of the file (and not parameters)
    #Then we cast to the right type
    for i in name_file.replace(".csv","").split("-")[1:]:
        try:
            parameters[i.split("=")[0]] = int(i.split("=")[1])
        except ValueError:
            try:
                parameters[i.split("=")[0]] = float(i.split("=")[1])
            except ValueError:
                if i.split("=")[1] == "True":
                    parameters[i.split("=")[0]] = True
                elif i.split("=")[1] == "False":
                    parameters[i.split("=")[0]] = False
                else:
                    parameters[i.split("=")[0]] = i.split("=")[1]
    return(parameters)

#To transform a dictionary of parameters to a name of file
def parameters_to_string(dictionary,parameter_not_included):
    for i in parameter_not_included:
        del(dictionary[i])
    return("-".join(str(elem) + "=" + str(key) for elem, key in dictionary.items()))

#To transform rules in separate values
def rules_matrix_to_list(action_rules,assessment_rules,type):
    res={}
    list_cases = ["11","10","01","00"]
    list_cases_simple = ["1","0"]
    res.update([("a_" + list_cases_simple[i] + "_" + type,rule) for i, rule in enumerate(action_rules)])
    res.update([("c_" + list_cases[i] + "_" + type, rule) for i, rule in enumerate(assessment_rules[:4])])
    res.update([("d_" + list_cases[i] + "_" + type, rule) for i, rule in enumerate(assessment_rules[4:])])
    return(res)


# To facilitate solving
def calculate_proba_assessment(assessment_rules):
    if (sym.ones(1,assessment_rules.shape[0])*assessment_rules)[0,0] == 4:
        proba_assessment = 1
    elif (sym.ones(1,assessment_rules.shape[0])*assessment_rules)[0,0] == 3:
        proba_assessment = 1 - (proba_matrix_second_order * (sym.ones(4, 1) - assessment_rules))[0, 0]
    else:
        proba_assessment  = (proba_matrix_second_order * assessment_rules)[0, 0]
    return(proba_assessment)

def calculate_proba_C(action_rules):
    proba_C = (proba_matrix_first_order * action_rules)[0,0]
    return proba_C

def calculate_proba_opinion_update_to_one(assessment_rules,proba_C):
    assessment_rules_C = assessment_rules.extract(range(0,4),[0])
    assessment_rules_D = assessment_rules.extract(range(4, 8),[0])
    proba_opinion_update_to_one = (proba_matrix_second_order * assessment_rules_C)[0, 0] * proba_C + (proba_matrix_second_order * assessment_rules_D)[0, 0] * (1-proba_C)
    return(sym.expand(proba_opinion_update_to_one))

def calculate_possible_rules(mirror,sample,seed):
    # All possible rules
    possible_action_rules = list(map(np.array, itertools.product([1, 0], repeat=2)))
    possible_assessment_rules = list(map(np.array, itertools.product([1, 0], repeat=8)))
    if mirror == True:
        #We still keep only one instance of AllC and AllD because the assessment rules are irrelevant
        possible_rules = list(map(list, itertools.product(
            [e for e in possible_action_rules if sum(e) != 0 and sum(e) != len(possible_action_rules[0])],
            possible_assessment_rules)))
        possible_rules.append([np.ones(len(possible_action_rules[0]), dtype=np.int32), np.ones(8, dtype=np.int32)])
        possible_rules.append([np.zeros(len(possible_action_rules[0]), dtype=np.int32), np.zeros(8, dtype=np.int32)])
    if mirror == False:
        # We keep one strategy as AllC and AllD because the different assessment rules are irrelevant
        possible_rules = list(map(list, itertools.product(
            [e for e in possible_action_rules if sum(e) != 0 and sum(e) != len(possible_action_rules[0])],
            possible_assessment_rules)))
        possible_rules.append([np.ones(len(possible_action_rules[0]), dtype=np.int32), np.ones(8, dtype=np.int32)])
        possible_rules.append([np.zeros(len(possible_action_rules[0]), dtype=np.int32), np.zeros(8, dtype=np.int32)])
        # We remove the mirror image if mirror image is False
        # We can remove in the loop because the mirror image is always after
        for i, rule_1 in enumerate(possible_rules):
            for j, rule_2 in enumerate(possible_rules):
                if (rule_2[0] == rule_1[0][::-1]).all() and (rule_2[1] == 1 - rule_1[1]).all():
                    possible_rules.pop(j)
            if i % 100 == 0:
                print("Removing mirror images : ", (i / len(possible_rules)))
    # If sample size provided, we sample number of strategies
    if sample != 0:
        random.seed(seed)
        possible_rules = random.sample(possible_rules, sample)
    return(possible_rules)

#---------------------------------------------------MODEL--------------------------------------------------------------
#Dynamic analysis: find equilibrium points (eq points) and stable points
def dynamic_analysis_continuous_time(differential_equa, variables, variables_ini,lower_bound,upper_bound):
    eq_points_theoretical = sym.solve(differential_equa,fast=True,manual = True,dict=True)
    #When one variable, output is not a list, when two variables, output is a list
    if type(eq_points_theoretical) != list:
        eq_points_theoretical=[eq_points_theoretical]
    eq_points=[]

    #It can happen that one variable is at equilibrium for any value but not the other
    #In this case, we replace the variable by its initial value and re-evaluate the eq points
    #This is an approximation and will differ from simulations because the variable will move from initial conditions the time that the second variable get to the equilibrium
    if eq_points_theoretical:
        for i in eq_points_theoretical:
            if any(i.get(v) == None for v in variables):
                for j, v in enumerate(variables):
                    if i.get(v) == None:
                        i.update({v: variables_ini[j]})
            #Re-evaluate for equilibrium equal to the other variable
            #In some cases, one variable is at equilibrium on the whole domain and the value of the other variable depends of this variable
                for v in variables:
                    if type(i[v]) != float:
                        i[v] = i[v].subs(i)
        # We identify the realisable equilibrium points which are not (i) complex numbers and (ii) inside of the definition domain [0,1]
        for i in eq_points_theoretical:
            if all(sym.re(j) >= lower_bound and sym.re(j) <= upper_bound and np.abs(sym.im(j.evalf())) < 0.001 for j in i.values()):
                set_eq_points= {k: (sym.re(v)).evalf() for k, v in i.items()}
                eq_points.append(set_eq_points)

    #Stability analysis
    stable_points=[]
    unstable_points=[]
    plateau = 0
    matrix_differential_equa = sym.Matrix(differential_equa)
    matrix_variable = sym.Matrix(variables)
    jacobian_matrix = matrix_differential_equa.jacobian(matrix_variable)
    eigenvalues = list(jacobian_matrix.eigenvals().keys())
    for i in eq_points:
        #For floating point error
        if all(sym.re(j.subs(i)) <= 0.00000001 for j in eigenvalues):
            stable_points.append(i)
            if all(sym.re(j.subs(i)) == 0 for j in eigenvalues):
                plateau = 1
        else:
            unstable_points.append(i)
    res={"eq_points_theoretical": eq_points_theoretical,
         "eq_points": eq_points,
         "stable_points": stable_points,
         "unstable_points": unstable_points,
         "N_stable_points": len(stable_points),
         "drift":0,
         "plateau":plateau}
    #Calculate the predicted value of each variable at equilibrium
    #When no stable points, either
    # (i) the equation is always positive -> go to upper boundary
    # (ii) the equation is always negative -> go to lower boundary
    # (iii) the equation is equal to 0 -> all points are equilibrium, we consider it stays at initial conditions
    if not stable_points:
        res["drift"] = 1
        for v_index, v in enumerate(variables):
            if differential_equa[v_index].subs([(e, variables_ini[v_index]) for e in variables]) > 0:
                res.update({v: upper_bound})
            elif differential_equa[v_index].subs([(e, variables_ini[v_index]) for e in variables]) < 0:
                res.update({v: lower_bound})
            else:
                res.update({v:variables_ini[v_index]})
    #When one stable point, the predicted value is the stable point
    elif len(stable_points) == 1:
        for v in variables:
            res.update({v: stable_points[0][v]})
    #If multiple stable points, we look at the direction of the difference equation at initial condition and find out the closest stable point
    elif len(stable_points)>1:
        for v_index, v in enumerate(variables):
            #We look at direction at initial conditions
            if np.abs(differential_equa[v_index].subs([(e, variables_ini[v_index]) for e in variables])) < 0.00000000001:
                closest_stable_point = variables_ini[v_index]
            elif differential_equa[v_index].subs([(e, variables_ini[v_index]) for e in variables]) < 0:
                closest_stable_point = lower_bound
                for i in stable_points:
                    #We ignore the stable points above the initial conditions
                    if i[v] > variables_ini[v_index]:
                        continue
                    elif np.abs(variables_ini[v_index] - i[v]) < np.abs(variables_ini[v_index] - closest_stable_point):
                        closest_stable_point = i[v]
            elif differential_equa[v_index].subs([(e, variables_ini[v_index]) for e in variables]) > 0:
                closest_stable_point = upper_bound
                for i in stable_points:
                    if i[v] < variables_ini[v_index]:
                        continue
                    elif np.abs(variables_ini[v_index] - i[v]) < np.abs(variables_ini[v_index] - closest_stable_point):
                        closest_stable_point = i[v]
            res.update({v: closest_stable_point})

    return(res)

#Analysis of a given strategy
def analysis(action_rules,assessment_rules, eps_a,eps_e, p_ini,plot):
    assessment_rules_C = assessment_rules.extract(range(0,4),[0])
    assessment_rules_D = assessment_rules.extract(range(4, 8),[0])
    # To facilitate solving
    assessment_if_C = calculate_proba_assessment(assessment_rules_C)
    assessment_if_D = calculate_proba_assessment(assessment_rules_D)

    variables=[p_ik]
    proba_C_no_error = calculate_proba_C(action_rules).subs([(p_i,p_ii),(p_j,p_ik)]).subs(p_j,p_ik)
    proba_C = (1 - eps_e) * proba_C_no_error + eps_e * (1 - proba_C_no_error)
    proba_o_no_error = assessment_if_C * proba_C + \
                       assessment_if_D * (1 - proba_C)
    proba_o = eps_a + (1 - 2 * eps_a) * proba_o_no_error

    differential_equa = proba_o - p_ik

    differential_equa = sym.expand(differential_equa.subs([(p_i,p_ik),(p_j,p_ik)]))

    res_equation = {"action_rules": "\"" +  "".join(str(elem) for elem in action_rules) + "\"",
                    "assessment_rules": "\"" + "".join(str(elem) for elem in assessment_rules) + "\"",
                    "rules": "\"" + "".join(str(elem) for elem in action_rules) + "".join(str(elem) for elem in assessment_rules) + "\"",
                    "action_rules_matrix": action_rules,
                    "assessment_rules_matrix": assessment_rules,
                    "differential_equa": differential_equa,
                    "proba C": proba_C,
    }

    #Dynamic analysis
    res_analysis = dynamic_analysis_continuous_time([differential_equa],variables,[p_ini]*len(variables),0,1)

    #Calculate cooperation probability
    for v in variables:
        proba_C = proba_C.subs(v,res_analysis[v])
    res_analysis.update({"cooperation": proba_C})

    #Fusion the two dictionaries
    res= {**res_equation, **res_analysis}
    if plot == True:
        lambda_differential_equa = sym.lambdify((p_ik), differential_equa)
        plt.title('Differential equation')
        plt.plot(np.arange(0, 1, 0.01), np.repeat(0, 100), np.arange(0, 1, 0.01),
                 np.vectorize(lambda_differential_equa)(np.arange(0, 1, 0.01)))
        plt.show()
    return(res)

#Analysis for a list of strategies
def sweep_analysis(write,sample,seed, eps_a,eps_e, p_ini):
    #Get parameters of the function and put it in the name file
    if write == True:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        parameters = parameters_to_string(values, ["write", "frame"])
        name_file = "analysis-" + parameters + ".csv"
    possible_rules = calculate_possible_rules(mirror=False,sample=sample,seed=seed)

    #Analysis for each rule
    list_res=[]
    for i, rule in enumerate(possible_rules):
            action_rules = sym.Matrix(rule[0])
            assessment_rules = sym.Matrix(rule[1])
            res = analysis(action_rules,assessment_rules, eps_a, eps_e,  p_ini,False)
            list_res.append(res)
            if i%20 == 0:
                print("Computing equations: ",i / (len(possible_rules)))
    dt = pd.DataFrame.from_dict(list_res)
    if write == True:
        dt.to_csv(name_file,index=False)
    else:
        return(dt)

#SIMULATIONS--------------------------------------------------------------------------------------------
#Detailed because we simulate explictly opinion (rather than reputation or just using the recursion equation)
def simulation_detailed(action_rules, assessment_rules,  N, N_o,  eps_a,eps_e, p_ini, N_gen, N_print, plot, detail):
    if eps_a == 0:
        eps_a = 0.0001
    if eps_e == 0:
        eps_e = 0.0001
    #Initialisation--------------------------------------------------------
    #Initialise the matrix of reputation
    M_opinions = np.ones((N,N),dtype=int)
    for i in range(N):
        for j in range(N):
            if random.random() > p_ini:
                M_opinions[i,j] = 0
    #Initialise the table of results
    p_ik_by_gen = np.zeros((N_gen-N_print, N))
    cooperation_by_gen = np.zeros(N_gen-N_print)
    #Reshape the matrix as explained at the beginning of the code
    action_rules_simul = np.flip(action_rules)
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))
    #Start simulation-----------------------------------------------------
    for i in range(N_gen):
        #Choose randomly donor and recipient
        actors = np.random.choice(N, 2+N_o, replace=False)
        #donor = actors[0]
        #recipient = actors[1]
        #observers = actors[2:]
        #Choose donor action as a function of their action rules and the reputations of donor and recipient
        donor_action = action_rules_simul[M_opinions[actors[0],actors[1]]]
        #Execution errors
        if random.random() < eps_e:
            donor_action = 1-donor_action
        #Writing action
        if i >= N_print:
            cooperation_by_gen[i-N_print] = donor_action

        #Opinion update
        for j in actors[2:]:
            #Update opinion with some probability as a function of assessment rules and the action of the donor
            M_opinions[j, actors[0]] = assessment_rules_simul[donor_action, M_opinions[j, actors[0]], M_opinions[j, actors[1]]]
            if random.random() < eps_a:
                M_opinions[j, actors[0]] = 1 - M_opinions[j, actors[0]]

        #Write the reputations
        if i >= N_print:
            p_ik_by_gen[i-N_print, :] = M_opinions[:, :].mean(axis=0) - M_opinions[:, :].diagonal() / N

    #End of simulation------------------------------------------
    if plot == True:
        plt.plot(range(N_print,N_gen),p_ik_by_gen[:,0],range(N_print,N_gen),p_ik_by_gen[:,1],range(N_print,N_gen),p_ik_by_gen[:,2],range(N_print,N_gen),p_ik_by_gen[:,3],
                  p_ik_by_gen[:,:].mean())
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.show()
    #Level of detail determinate the output (either by generation or mean across reputation)
    if detail == 1:
        res = {"p_ik_by_gen": p_ik_by_gen, "cooperation_by_gen": cooperation_by_gen}
    if detail == 0:
        res = {"p_ik": p_ik_by_gen[:, :].mean(), "sd_p_ik": p_ik_by_gen[:, :].std(),
                "cooperation": cooperation_by_gen[:].mean(),"std_cooperation": cooperation_by_gen[:].std()}
    return(res)


def simulations_detailed_replicated(action_rules, assessment_rules,  N, N_o, eps_a,eps_e, p_ini,  N_gen, N_print, N_simul):
    list_res=[]
    for i in range(N_simul):
        res_single_run = simulation_detailed(action_rules=action_rules, assessment_rules = assessment_rules,
                                                          N=N, N_o=N_o, eps_a=eps_a,eps_e=eps_e, p_ini=p_ini,
                                                          N_gen=N_gen, N_print=N_print, plot=False, detail=0)
        list_res.append(res_single_run)
    dt_by_simul=pd.DataFrame.from_dict(list_res)
    dt_mean= dt_by_simul.mean().add_prefix("mean_simul_")
    dt_se = dt_by_simul.sem().add_prefix("se_simul_")
    res = pd.concat([dt_mean,dt_se]).to_dict()
    return(res)

#Simulations for list of strategies
def sweep_simulations(input_file,  N, N_o, N_gen, N_print, N_simul, write):
        #Read a file with analytical result and get the parameters from it
        dt_analytic = pd.read_csv(input_file)
        parameters_input_file=read_parameters(input_file)
        list_res=[]
        #Simulations for each strategies
        #pool = mp.Pool(7)
        for index, row in dt_analytic.iterrows():
            #list_res.append(pool.apply(simulations_detailed_replicated, args=(np.array([int(i) for i in row["action_rules"].replace("\"", "")]),
            #                                              np.array([int(i) for i in row["assessment_rules"].replace("\"", "")]),
            #                                               N,   N_o,
            #                                              parameters_input_file["eps_a"],  parameters_input_file["eps_e"],parameters_input_file["p_ini"],
            #                                              N_gen,N_print,N_simul)))
            simulations = simulations_detailed_replicated(action_rules=np.array([int(i) for i in row["action_rules"].replace("\"", "")]),
                                                          assessment_rules=np.array([int(i) for i in row["assessment_rules"].replace("\"", "")]),
                                                          N = N, N_o =  N_o,
                                                          eps_a=parameters_input_file["eps_a"], eps_e= parameters_input_file["eps_e"], p_ini=parameters_input_file["p_ini"],
                                                          N_gen=N_gen,N_print=N_print,N_simul=N_simul)

            simulations.update({"diff_p_ik": np.abs(row["p_ik"] - simulations["mean_simul_p_ik"])})
            simulations.update({"diff_coop": np.abs(row["cooperation"] - simulations["mean_simul_cooperation"])})
            list_res.append(simulations)
            print(index / len(dt_analytic.index))
        #pool.close()
        dt_simul=pd.DataFrame.from_dict(list_res)
        dt_res = pd.concat((dt_analytic,dt_simul),axis=1)
        if write == True:
            name_file = "simul-" + parameters_to_string(parameters_input_file,[]) + "-gen=" + str(N_gen/1000) + "k" + "-print=" + str(N_print/1000) + "k" + "-S=" + str(N_simul) + ".csv"
            dt_res.to_csv(name_file,index=False)
        return(dt_res)

#---------------------------------------------Evolutionary dynamics---------------------------------------------------
def ESS_analysis(action_rules_r,assessment_rules_r,p_rr,action_rules_m, assessment_rules_m,eps_a,eps_e,p_ini):
    #Get it from analysis or file directly
    #Be careful, variables need to be in the same order than equations
    proba_C_m = calculate_proba_C(action_rules_m)
    proba_C_r = calculate_proba_C(action_rules_r)
    proba_C_m = (1-eps_e)*proba_C_m + eps_e*(1-proba_C_m)
    proba_C_r = (1-eps_e)*proba_C_r + eps_e*(1-proba_C_r)

    assessment_rules_C_m = assessment_rules_m.extract(range(0,4),[0])
    assessment_rules_D_m = assessment_rules_m.extract(range(4, 8),[0])
    assessment_if_C_m = calculate_proba_assessment(assessment_rules_C_m)
    assessment_if_D_m = calculate_proba_assessment(assessment_rules_D_m)

    assessment_rules_C_r = assessment_rules_r.extract(range(0,4),[0])
    assessment_rules_D_r = assessment_rules_r.extract(range(4, 8),[0])
    assessment_if_C_r = calculate_proba_assessment(assessment_rules_C_r)
    assessment_if_D_r = calculate_proba_assessment(assessment_rules_D_r)


    #We could solve this one directly
    proba_o_no_error_rm = assessment_if_C_m * proba_C_r.subs(p_j,p_rr) + \
                   assessment_if_D_m * (1 - proba_C_r.subs(p_j,p_rr))
    proba_o_rm = eps_a + (1 - 2 * eps_a) * proba_o_no_error_rm
    proba_o_rm = proba_o_rm.subs([(p_i, p_rm), (p_j, p_rm)])
    differential_equa_rm = proba_o_rm - p_rm

    analysis_p_rm = dynamic_analysis_continuous_time([differential_equa_rm], [p_rm], [p_ini], 0, 1)

    # Calcul of equation of resident on mutant p_mr
    proba_o_no_error_mr = assessment_if_C_r * proba_C_m.subs(p_j,analysis_p_rm[p_rm]) + \
                   assessment_if_D_r * (1 - proba_C_m.subs(p_j,analysis_p_rm[p_rm]))
    proba_o_mr = eps_a + (1 - 2 * eps_a) * proba_o_no_error_mr
    proba_o_mr = proba_o_mr.subs([(p_i, p_mr), (p_j, p_rr)])
    differential_equa_mr = proba_o_mr - p_mr
    analysis_p_mr = dynamic_analysis_continuous_time([differential_equa_mr], [p_mr], [p_ini], 0, 1)

    #analysis = dynamic_analysis(difference_equa, variables, [p_ini] * len(variables), 0, 1)
    #Fitness is the proportion of cooperation toward an individual
    proba_C_mr = proba_C_r.subs(p_j,analysis_p_mr[p_mr])
    proba_C_rm = proba_C_m.subs(p_j,analysis_p_rm[p_rm])
    proba_C_rr = proba_C_r.subs(p_j, p_rr)
    fitness_m = b *proba_C_mr - c *proba_C_rm
    fitness_r = b *proba_C_rr - c * proba_C_rr


    res = {"action_rules_r": "\"" +  "".join(str(elem) for elem in action_rules_r) + "\"",
            "assessment_rules_r": "\"" +  "".join(str(elem) for elem in assessment_rules_r) + "\"",
            "rules_r": "\"" +  "".join(str(elem) for elem in action_rules_r) + "".join(str(elem) for elem in assessment_rules_r) + "\"",
            "action_rules_m": "\"" +  "".join(str(elem) for elem in action_rules_m) + "\"",
            "assessment_rules_m": "\"" +  "".join(str(elem) for elem in assessment_rules_m) + "\"",
            "rules_m": "\"" + "".join(str(elem) for elem in action_rules_m) + "".join(str(elem) for elem in assessment_rules_m) + "\"",
            "differential_equa_rm": differential_equa_rm, "differential_equa_mr": differential_equa_mr,
            "p_rr": p_rr, "p_mr": analysis_p_mr[p_mr], "p_rm": analysis_p_rm[p_rm],
            "proba_C_rr": proba_C_rr, "proba_C_mr": proba_C_mr, "proba_C_rm": proba_C_rm,
            "fitness_r":fitness_r, "fitness_m": fitness_m,"diff_fitness": sym.expand(fitness_m - fitness_r)}
    #To do correlations
    res= {**res, **rules_matrix_to_list(action_rules_r,assessment_rules_r,"r"),**rules_matrix_to_list(action_rules_m,assessment_rules_m,"m")}
    return(res)


def sweep_ESS_analysis(input_file):
    dt_analytic = pd.read_csv(input_file)
    parameters_input_file=read_parameters(input_file)
    list_res=[]
    for index_r, row_r in dt_analytic.iterrows():
        for index_m, row_m in dt_analytic.iterrows():
            # If same strategies
            #if index_r == index_m:
            #    continue
            res=ESS_analysis(action_rules_r=parse_expr(row_r["action_rules_matrix"]),
                               assessment_rules_r=parse_expr(row_r["assessment_rules_matrix"]),
                               p_rr = row_r["p_ik"],
                               action_rules_m = parse_expr(row_m["action_rules_matrix"]),
                               assessment_rules_m= parse_expr(row_m["assessment_rules_matrix"]),
                               eps_a = parameters_input_file["eps_a"],eps_e = parameters_input_file["eps_e"],
                               p_ini = parameters_input_file["p_ini"])
            list_res.append(res)
        print(index_r/len(dt_analytic.index))
    dt = pd.DataFrame.from_dict(list_res)
    name_file = "ESS-" + parameters_to_string(parameters_input_file, []) + ".csv"
    dt.to_csv(name_file,index=False)

def simulation_ESS(action_rules_r, assessment_rules_r,action_rules_m, assessment_rules_m,  N, N_o,  eps_a,eps_e, p_ini, N_gen, N_print, plot, detail):
    if eps_a == 0:
        eps_a = 0.0001
    if eps_e == 0:
        eps_e = 0.0001
    #Initialisation--------------------------------------------------------
    #Initialise the matrix of reputation
    M_opinions = np.ones((N,N),dtype=int)
    for i in range(N):
        for j in range(N):
            if random.random() > p_ini:
                M_opinions[i,j] = 0
    #Initialise the table of results
    p_rr_by_gen = np.zeros((N_gen-N_print, N-1))
    p_rm_by_gen = np.zeros((N_gen - N_print, 1))
    p_mr_by_gen = np.zeros((N_gen - N_print, 1))
    cooperation_rr_by_gen = []
    cooperation_rm_by_gen = []
    cooperation_mr_by_gen = []
    #Reshape the matrix as explained at the beginning of the code
    action_rules_r_simul = np.flip(action_rules_r)
    assessment_rules_r_simul = np.reshape(np.flip(assessment_rules_r),(2,2,2))
    action_rules_m_simul = np.flip(action_rules_m)
    assessment_rules_m_simul = np.reshape(np.flip(assessment_rules_m), (2, 2, 2))
    #Start simulation-----------------------------------------------------
    for i in range(N_gen):
        #Choose randomly donor and recipient
        actors = np.random.choice(N, 2+N_o, replace=False)
        donor = actors[0]
        recipient = actors[1]
        observers = actors[2:]
        #Choose donor action as a function of their action rules and the reputations of donor and recipient
        if donor == 0:
            donor_action = action_rules_m_simul[M_opinions[donor, recipient]]
        else:
            donor_action = action_rules_r_simul[M_opinions[donor, recipient]]

        #Execution errors
        if random.random() < eps_e:
            donor_action = 1-donor_action
        #Writing action
        if i >= N_print:
            if donor == 0:
                cooperation_rm_by_gen.append(donor_action)
            elif recipient == 0:
                cooperation_mr_by_gen.append(donor_action)
            else:
                cooperation_rr_by_gen.append(donor_action)

        #Opinion update
        for j in observers:
            #Update opinion with some probability as a function of assessment rules and the action of the donor
            if j == 0:
                M_opinions[j, donor] = assessment_rules_m_simul[donor_action, M_opinions[j, donor], M_opinions[j, recipient]]
            else:
                M_opinions[j, donor] = assessment_rules_r_simul[donor_action, M_opinions[j, donor], M_opinions[j, recipient]]
            if random.random() < eps_a:
                M_opinions[j, donor] = 1 - M_opinions[j, donor]

        #Write the reputations
        if i >= N_print:
            p_rr_by_gen[i-N_print, :] = M_opinions[1:, 1:].mean(axis=0) - M_opinions[1:, 1:].diagonal() / N
            p_rm_by_gen[i-N_print,0] = M_opinions[0, 1:].mean(axis=0)
            p_mr_by_gen[i - N_print, 0] = M_opinions[1:, 0].mean(axis=0)
    #End of simulation------------------------------------------
    if plot == True:
        plt.plot(range(N_print,N_gen),p_rm_by_gen[:,0],range(N_print,N_gen),p_mr_by_gen[:,0],
                  p_rr_by_gen[:,:].mean())
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.show()
    #Level of detail determinate the output (either by generation or mean across reputation)
    if detail == 1:
        res = {"p_rr_by_gen": p_rr_by_gen,"p_rm_by_gen": p_rm_by_gen,"p_mr_by_gen": p_mr_by_gen,
               "cooperation_rr_by_gen": cooperation_rr_by_gen, "cooperation_rm_by_gen": cooperation_rm_by_gen, "cooperation_mr_by_gen": cooperation_mr_by_gen}
    if detail == 0:
        res = {"p_rr": p_rr_by_gen[:, :].mean(), "sd_p_rr": p_rr_by_gen[:, :].std(),
               "p_rm": p_rm_by_gen[:, :].mean(), "sd_p_rm": p_rm_by_gen[:, :].std(),
               "p_mr": p_mr_by_gen[:, :].mean(), "sd_p_mr": p_mr_by_gen[:, :].std(),
                "cooperation_rr": cooperation_rr_by_gen[:].mean(),"std_cooperation_rr": cooperation_rr_by_gen[:].std(),
               "cooperation_rm": cooperation_rm_by_gen[:].mean(),"std_cooperation_rm": cooperation_rm_by_gen[:].std(),
               "cooperation_mr": cooperation_mr_by_gen[:].mean(),"std_cooperation_mr": cooperation_mr_by_gen[:].std()}
    return(res)













###==================================================ANALYSIS=======================================================================
#----------------------------------For particular strategy------------------------------------------------

# print(analysis(action_rules=sym.Matrix([1,0]),
#              assessment_rules=sym.Matrix([1,1,1,0,0,0,1,0]),
#              eps_a=0.05,
#              eps_e=0.05,
#              p_ini=0.1,
#              plot=True))

# simulation_detailed(action_rules=np.array([1,0]), assessment_rules=np.array([1,1,1,0,0,0,1,0]),
#                    N = 100, N_o = 1,
#                    eps_a = 0.05, eps_e = 0.05,
#                    p_ini = 0.1,
#                    N_gen = 1000000,N_print = 0 , plot = True, detail = 0)

#-------------------------------------------------------------simulation of single run-----------------------------------------------

# dt_print_simul=pd.DataFrame.from_dict(simulation_detailed(action_rules=np.array([1,0]), assessment_rules=np.array([1,1,0,0,0,1,1,1]),
#                     N = 100, N_o = 1,
#                     eps_a = 0.0, eps_e = 0.0,
#                     p_ini = 0.5,
#                     N_gen = 500000,N_print = 0 , plot = False, detail = 1)["p_ik_by_gen"])
# dt_print_simul.to_csv("simul_single_run_11000111.csv")
#
# dt_print_simul=pd.DataFrame.from_dict(simulation_detailed(action_rules=np.array([1,0]), assessment_rules=np.array([1,1,1,1,0,0,0,0]),
#                     N = 100, N_o = 1,
#                     eps_a = 0.0, eps_e = 0.0,
#                     p_ini = 0.5,
#                     N_gen = 500000,N_print = 0 , plot = False, detail = 1)["p_ik_by_gen"])
# dt_print_simul.to_csv("simul_single_run_11110000.csv")
#
# dt_print_simul=pd.DataFrame.from_dict(simulation_detailed(action_rules=np.array([1,0]), assessment_rules=np.array([1,1,1,1,0,0,0,1]),
#                     N = 100, N_o = 1,
#                     eps_a = 0.0, eps_e = 0.0,
#                     p_ini = 0.5,
#                     N_gen = 500000,N_print = 0 , plot = False, detail = 1)["p_ik_by_gen"])
# dt_print_simul.to_csv("simul_single_run_11110001.csv")

#-----------------------ESS analysis for the ESS---------------------------------------------------------
# list_ESS = [sym.Matrix([1,1,1,1,0,1,0,1]),sym.Matrix([1,1,1,1,0,1,0,0]), sym.Matrix([1,1,1,1,0,0,0,1]),sym.Matrix([1,1,1,0,0,1,0,1]),
#             sym.Matrix([1,1,1,1,0,1,1,1]), sym.Matrix([1,1,1,1,0,1,1,0]),sym.Matrix([1,1,1,1,0,0,1,1]), sym.Matrix([1,1,1,1,0,0,1,0]),
#             sym.Matrix([1,1,1,0,0,1,1,1]),sym.Matrix([1,1,1,0,0,1,1,0]), sym.Matrix([1,1,1,0,0,0,1,1]),sym.Matrix([1,1,0,1,1,1,0,1]),
#             sym.Matrix([1,1,0,1,1,1,0,0]),sym.Matrix([1,1,0,1,1,0,0,1]), sym.Matrix([1,1,0,0,1,1,0,1])]
# list_res=[]
# for i in list_ESS:
#     res_analysis= ESS_analysis(action_rules_r=sym.Matrix([1,0]),assessment_rules_r=i,p_rr=1,
#              action_rules_m=sym.Matrix([0,0]),assessment_rules_m=sym.Matrix([0,0,0,0,0,0,0,0]),
#              eps_a = 0, eps_e = 0, p_ini = 0.5)
#     res = {"rules_r": res_analysis["rules_r"], "differential_equa_mr": res_analysis["differential_equa_mr"]}
#     list_res.append(res)
# dt_print=pd.DataFrame.from_dict(list_res)
# dt_print.to_csv("Equa_ESS_coop.csv")

#----------------------To compare the results across different values of parameterrs-----------------------------------------

# for i in [0.01,0.1,0.5]:
#     for j in [0,0.01,0.1,]:
#         print("q =",i, end = ' ')
#         print("epsilon = ", j , end = ' ')
#         print(simulation_detailed(N=100,N_gen=800000,q=i,epsilon=j,p_ini=0.5,
#             action_rules = np.array([0,1,0,0]),assessment_rules=np.array([0,1,0,0,1,1,1,0]),donor_obs=True,plot=False,error="action",detail=0,N_print=400000)["p_ik"]-
#               analysis(sym.Matrix([0,1,0,0]),sym.Matrix([0,1,0,0,1,1,1,0]),True,i,j,j,100,0.5)[p_ik])

#---------------------------------Analysis for list of strategies-----------------------------------------------

# sweep_analysis(write=True,sample=0,seed=1,eps_a=0.05,eps_e=0.05,p_ini=0.1)
#
# sweep_analysis(write=True,sample=0,seed=1,eps_a=0.05,eps_e=0.05,p_ini=0.5)
#
# sweep_analysis(write=True,sample=0,seed=1,eps_a=0.05,eps_e=0.05,p_ini=0.9)


#---------------------------------------For simulations across a list of strategies-----------------------------------------
start_time = time.time()
print(str(start_time))
input_file = "analysis-sample=0-seed=1-eps_a=0.0-eps_e=0.0-p_ini=0.1.csv"
#sweep_simulations(input_file=input_file, N = 100,N_o = 1, N_gen=500000, N_print=400000, N_simul=30, write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=0-seed=1-eps_a=0.0-eps_e=0.0-p_ini=0.5.csv"
#sweep_simulations(input_file=input_file, N = 100,N_o = 1, N_gen=500000, N_print=400000, N_simul=30, write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=0-seed=1-eps_a=0.0-eps_e=0.0-p_ini=0.9.csv"
#sweep_simulations(input_file=input_file, N = 100,N_o = 1, N_gen=500000, N_print=400000, N_simul=30, write=True)
print("Execution time = " + str((time.time() - start_time)/60))

input_file = "analysis-sample=0-seed=1-eps_a=0.05-eps_e=0.05-p_ini=0.1.csv"
#sweep_simulations(input_file=input_file, N = 100,N_o = 1, N_gen=500000, N_print=400000, N_simul=30, write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=0-seed=1-eps_a=0.05-eps_e=0.05-p_ini=0.5.csv"
sweep_simulations(input_file=input_file, N = 100,N_o = 1, N_gen=500000, N_print=400000, N_simul=30, write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=0-seed=1-eps_a=0.05-eps_e=0.05-p_ini=0.9.csv"
#sweep_simulations(input_file=input_file, N = 100,N_o = 1, N_gen=500000, N_print=400000, N_simul=30, write=True)
print("Execution time = " + str((time.time() - start_time)/60))


#------------------------------------------ESS analysis on list of strategies----------------------------------------------
input_file = "analysis-sample=0-seed=1-eps_a=0.05-eps_e=0.05-p_ini=0.9.csv"
#sweep_ESS_analysis(input_file = input_file)
print("Execution time = " + str((time.time() - start_time)/60))

input_file = "analysis-sample=0-seed=1-eps_a=0.05-eps_e=0.05-p_ini=0.9.csv"
#sweep_ESS_analysis(input_file = input_file)
print("Execution time = " + str((time.time() - start_time)/60))

input_file = "analysis-sample=0-seed=1-eps_a=0.05-eps_e=0.05-p_ini=0.9.csv"
#sweep_ESS_analysis(input_file = input_file)
print("Execution time = " + str((time.time() - start_time)/60))