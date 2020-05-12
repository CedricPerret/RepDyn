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

#List of pre-defined strategies:
#The matrix are flipped for simulations. This is because the custom is to start the matrix from cooperation to defection,
# but the custom in code is to start from defection (defection = 0) to cooperation (cooperation = 1)

class Individual(metaclass=abc.ABCMeta):
    def __init__(self):
        self.action = None
        self.reputation = 1

class AllD(Individual):
    action_rules = sym.Matrix([0, 0, 0, 0])

    action_rules_simul = np.reshape(action_rules,(2,2))

class Leading_one(Individual):
    assessment_rules = sym.Matrix([1, 1, 1, 1, 0, 1, 0, 0])
    action_rules = sym.Matrix([1, 0, 1, 1])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

class Leading_two(Individual):
    assessment_rules = sym.Matrix([1, 0, 1, 1, 0, 1, 0, 0])
    action_rules = sym.Matrix([1, 0, 1, 1])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))
class Leading_three(Individual):
    assessment_rules = sym.Matrix([1, 1, 1, 1, 0, 1, 0, 1])
    action_rules = sym.Matrix([1, 0, 1, 0])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

class Leading_four(Individual):
    assessment_rules = sym.Matrix([1, 1, 1, 0, 0, 1, 0, 1])
    action_rules = sym.Matrix([1, 0, 1, 0])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

class Leading_five(Individual):
    assessment_rules = sym.Matrix([1, 0, 1, 1, 0, 1, 0, 1])
    action_rules = sym.Matrix([1, 0, 1, 0])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

class Leading_six(Individual):
    assessment_rules = sym.Matrix([1, 0, 1, 0, 0, 1, 0, 1])
    action_rules = sym.Matrix([1, 0, 1, 0])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

class Leading_seven(Individual):
    assessment_rules = sym.Matrix([1, 1, 1, 0, 0, 1, 0, 0])
    action_rules = sym.Matrix([1, 0, 1, 0])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

class Leading_eight(Individual):
    assessment_rules = sym.Matrix([1, 0, 1, 0, 0, 1, 0, 0])
    action_rules = sym.Matrix([1, 0, 1, 0])

    action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))

#List of symbols
#Reputations
p_i = sym.Symbol('p_i')
p_ii = sym.Symbol('p_ii')
p_ik = sym.Symbol('p_ik')
p_j = sym.Symbol('p_j')
p_mr = sym.Symbol('p_mr')
p_rm = sym.Symbol('p_rm')

#Parameters
#Error rate
epsilon = sym.Symbol("epsilon")
#Proportion of observers
q = sym.Symbol("q")

#Generic symbol, in case it is needed
x = sym.Symbol('x')


#Probabilities tree that describes the probability of each encounter: 11, 10, 01, 00 or 1,0
proba_tree = sym.Matrix([[p_i * p_j, p_i * (1 - p_j), (1 - p_i) * p_j, (1 - p_i) * (1 - p_j)]])
proba_tree_simple = sym.Matrix([[p_j, (1 - p_j)]])

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
        proba_assessment = 1 - (proba_tree * (sym.ones(4, 1) - assessment_rules))[0, 0]
    else:
        proba_assessment  = (proba_tree * assessment_rules)[0, 0]
    return(proba_assessment)

def calculate_proba_C(action_rules):
    if len(action_rules) == 4:
        proba_C = (proba_tree * action_rules)[0,0]
    elif len(action_rules) ==2:
        proba_C = (proba_tree_simple * action_rules)[0,0]
    return proba_C

def calculate_freq_G_in_observers(assessment_rules,proba_C):
    assessment_rules_C = assessment_rules.extract(range(0,4),[0])
    assessment_rules_D = assessment_rules.extract(range(4, 8),[0])
    p_D_prime = (proba_tree * assessment_rules_C)[0, 0] * proba_C + (proba_tree * assessment_rules_D)[0, 0] * (1-proba_C)
    return(sym.expand(p_D_prime))

def calculate_possible_rules(mirror,donor_obs,sample,seed):
    # All possible rules
    if donor_obs == True:
        possible_action_rules = list(map(np.array, itertools.product([1, 0], repeat=4)))
    elif donor_obs == False:
        possible_action_rules = list(map(np.array, itertools.product([1, 0], repeat=2)))
    possible_assessment_rules = list(map(np.array, itertools.product([1, 0], repeat=8)))

    if mirror == True:
        possible_rules = list(map(list, itertools.product(possible_action_rules, possible_assessment_rules)))
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

#Dynamic analysis: find equilibrium points (eq points) and stable points
def dynamic_analysis(difference_equations, variables, variables_ini,lower_bound,upper_bound):

    eq_points_theoretical = sym.solve(difference_equations,fast=True,manual = True,dict=True)
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
            if all(sym.re(j) >= lower_bound and sym.re(j) <= upper_bound and np.abs(sym.im(j)) < 0.001 for j in i.values()):
                set_eq_points= {k: sym.re(v) for k, v in i.items()}
                eq_points.append(set_eq_points)

    #Stability analysis
    stable_points=[]
    unstable_points=[]
    matrix_recursion_equa = sym.Matrix([equa+variables[index] for index,equa in enumerate(difference_equations)])
    matrix_variable = sym.Matrix(variables)
    jacobian_matrix = matrix_recursion_equa.jacobian(matrix_variable)
    eigenvalues = list(jacobian_matrix.eigenvals().keys())
    for i in eq_points:
        #For floating point error
        if all(np.abs(sym.sqrt(sym.re(j.subs(i)) ** 2 + sym.im(j.subs(i)) ** 2)) <= 1.00001 for j in eigenvalues):
            stable_points.append(i)
        else:
            unstable_points.append(i)
    res={"eq_points_theoretical": eq_points_theoretical,
         "eq_points": eq_points,
         "stable_points": stable_points,
         "unstable_points": unstable_points,
         "N_stable_points": len(stable_points)}
    #Calculate the predicted value of each variable at equilibrium
    #When no stable points, either
    # (i) the equation is always positive -> go to upper boundary
    # (ii) the equation is always negative -> go to lower boundary
    # (iii) the equation is equal to 0 -> all points are equilibrium, we consider it stays at initial conditions
    if not stable_points:
        for v_index, v in enumerate(variables):
            if difference_equations[v_index].subs([(e, variables_ini[v_index]) for e in variables]) > 0:
                res.update({v: upper_bound})
            elif difference_equations[v_index].subs([(e, variables_ini[v_index]) for e in variables]) < 0:
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
            if difference_equations[v_index].subs([(e, variables_ini[v_index]) for e in variables]) < 0.00001:
                closest_stable_point = 0.5

            elif difference_equations[v_index].subs([(e, variables_ini[v_index]) for e in variables]) < 0:
                closest_stable_point = lower_bound
                for i in stable_points:
                    #We ignore the stable points above the initial conditions
                    if i[v] > variables_ini[v_index]:
                        continue
                    elif np.abs(variables_ini[v_index] - i[v]) < np.abs(variables_ini[v_index] - closest_stable_point):
                        closest_stable_point = i[v]
            elif difference_equations[v_index].subs([(e, variables_ini[v_index]) for e in variables]) > 0:
                closest_stable_point = upper_bound
                for i in stable_points:
                    if i[v] < variables_ini[v_index]:
                        continue
                    elif np.abs(variables_ini[v_index] - i[v]) < np.abs(variables_ini[v_index] - closest_stable_point):
                        closest_stable_point = i[v]
            res.update({v: closest_stable_point})

    return(res)


#Analysis of a given strategy
def analysis(action_rules,assessment_rules,donor_obs, error, q, epsilon ,N, p_ini):
    assessment_rules_C = assessment_rules.extract(range(0,4),[0])
    assessment_rules_D = assessment_rules.extract(range(4, 8),[0])
    # To facilitate solving
    assessment_if_C = calculate_proba_assessment(assessment_rules_C)
    assessment_if_D = calculate_proba_assessment(assessment_rules_D)

    if donor_obs == True:
        variables = [p_ii,p_ik]
        #Calculate equation for p_ii
        if error == "assessment" or error == "both":
            assessment_rules_ii = sym.matrix_multiply_elementwise(action_rules, assessment_rules_C) + sym.matrix_multiply_elementwise((sym.ones(4, 1) - action_rules), assessment_rules_D)
            freq_G_in_ii = sym.expand((proba_tree * assessment_rules_ii)[0, 0])
        if error == "action" or error == "both":
            assessment_rules_ii = sym.matrix_multiply_elementwise(sym.matrix_multiply_elementwise(action_rules,assessment_rules_C),sym.Matrix([1-epsilon,1-epsilon,1-epsilon,1-epsilon])) +\
                                  sym.matrix_multiply_elementwise(sym.matrix_multiply_elementwise(action_rules,assessment_rules_D),sym.Matrix([epsilon,epsilon,epsilon,epsilon]))+ \
                                  sym.matrix_multiply_elementwise(sym.matrix_multiply_elementwise((sym.ones(4, 1) - action_rules), assessment_rules_D),sym.Matrix([1-epsilon,1-epsilon,1-epsilon,1-epsilon]))+\
                                  sym.matrix_multiply_elementwise(sym.matrix_multiply_elementwise((sym.ones(4, 1) - action_rules), assessment_rules_C),sym.Matrix([epsilon,epsilon,epsilon,epsilon]))
            freq_G_in_ii = sym.expand((proba_tree * assessment_rules_ii)[0, 0])

        if error == "assessment" or error == "both":
            freq_G_in_ii = freq_G_in_ii * (1-epsilon) + \
                            epsilon * (1-p_i)

        freq_G_in_ii = sym.expand(freq_G_in_ii.subs([(p_i,p_ii),(p_j,p_ik)]))
        recursion_equa_p_ii = p_ii + q * (freq_G_in_ii - p_ii)
        difference_equa_p_ii = recursion_equa_p_ii - p_ii
        #Calculate equation for p_ik
        proba_C = calculate_proba_C(action_rules).subs([(p_i,p_ii),(p_j,p_ik)])
        if error == "action" or error == "both":
            proba_C = (1-epsilon)*proba_C + epsilon * (1-proba_C) - epsilon*proba_C
        freq_G_in_ik = assessment_if_C * proba_C + assessment_if_D * (1 - proba_C)
        if error == "assessment" or error == "both":
            freq_G_in_ik = freq_G_in_ik*(1-epsilon) + epsilon * (1-p_i)
        freq_G_in_ik = sym.expand(freq_G_in_ik.subs([(p_i,p_ik),(p_j,p_ik)]))
        recursion_equa_p_ik= p_ik + q*(freq_G_in_ik - p_ik)
        difference_equa_p_ik = recursion_equa_p_ik - p_ik

        recursion_equa=[sym.expand(recursion_equa_p_ii),sym.expand(recursion_equa_p_ik)]
        difference_equa=[sym.expand(difference_equa_p_ii),sym.expand(difference_equa_p_ik)]

    elif donor_obs == False:
        variables=[p_ik]
        proba_C = calculate_proba_C(action_rules).subs([(p_i,p_ii),(p_j,p_ik)]).subs(p_j,p_ik)
        if error == "action" or error == "both":
            proba_C = (1 - epsilon) * proba_C + epsilon * (1 - proba_C) - epsilon * proba_C
        freq_G_in_ik = assessment_if_C * proba_C + \
                       assessment_if_D * (1 - proba_C)
        if error == "assessment" or error == "both":
            freq_G_in_ik = freq_G_in_ik*(1-epsilon) + epsilon * (1-p_i)

        freq_G_in_ik = sym.expand(freq_G_in_ik.subs([(p_i,p_ik),(p_j,p_ik)]))
        recursion_equa_p_ik = p_ik + q*(freq_G_in_ik - p_ik)
        difference_equa_p_ik = recursion_equa_p_ik - p_ik
        recursion_equa = [sym.expand(recursion_equa_p_ik)]
        difference_equa = [sym.expand(difference_equa_p_ik)]

    res_equation = {"action_rules": "\"" +  "".join(str(elem) for elem in action_rules) + "\"",
                    "assessment_rules": "\"" + "".join(str(elem) for elem in assessment_rules) + "\"",
                    "action_rules_matrix": action_rules,
                    "assessment_rules_matrix": assessment_rules,
                    "recursion_equa": recursion_equa,
                    "difference_equa": difference_equa,
                    "proba C": proba_C
    }

    #Dynamic analysis
    res_analysis = dynamic_analysis(difference_equa,variables,[p_ini]*len(variables),0,1)
    #If donor is an observer, we need to calculate the average reputation with p_ii and p_ik
    if donor_obs == True:
        res_analysis.update({"p": (1/N)*res_analysis[p_ii] + ((N-1)/N) * res_analysis[p_ik]})

    #Calculate cooperation probability
    for v in variables:
        proba_C = proba_C.subs(v,res_analysis[v])
    res_analysis.update({"cooperation": proba_C})

    #Fusion the two dictionaries
    res= {**res_equation, **res_analysis}
    return(res)

#Analysis for a list of strategies
def sweep_rules_analysis(write,mirror,sample,seed,donor_obs, error, q , epsilon,N, p_ini):
    #Get parameters of the function and put it in the name file
    if write == True:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        parameters = parameters_to_string(values, ["write", "frame","mirror"])
        name_file = "analysis-" + parameters + ".csv"
    possible_rules = calculate_possible_rules(mirror,donor_obs,sample,seed)

    #Analysis for each rule
    list_res=[]
    for i, rule in enumerate(possible_rules):
            action_rules = sym.Matrix(rule[0])
            assessment_rules = sym.Matrix(rule[1])
            res = analysis(action_rules,assessment_rules,donor_obs,error, q , epsilon,N,  p_ini)
            list_res.append(res)
            if i%20 == 0:
                print("Computing equations: ",i / (len(possible_rules)))
    dt = pd.DataFrame.from_dict(list_res)
    if write == True:
        dt.to_csv(name_file)
    else:
        return(dt)

#---------------------------------------------Evolutionary dynamics---------------------------------------------------
def ESS_analysis(action_rules_r,assessment_rules_r,p_rr,action_rules_m, assessment_rules_m,donor_obs,error,q,epsilon,p_ini,benefit,cost):
    #Get it from analysis or file directly
    #Be careful, variables need to be in the same order than equations
    variables = [p_mr,p_rm]
    proba_C_m = calculate_proba_C(action_rules_m)
    proba_C_r = calculate_proba_C(action_rules_r)
    if error == "action" or error == "both":
        proba_C_m = (1-epsilon)*proba_C_m + epsilon*(1-proba_C_m) - epsilon*proba_C_m
        proba_C_r = (1-epsilon)*proba_C_r + epsilon*(1-proba_C_r) - epsilon*proba_C_r

    assessment_rules_C_m = assessment_rules_m.extract(range(0,4),[0])
    assessment_rules_D_m = assessment_rules_m.extract(range(4, 8),[0])
    assessment_if_C_m = calculate_proba_assessment(assessment_rules_C_m)
    assessment_if_D_m = calculate_proba_assessment(assessment_rules_D_m)

    assessment_rules_C_r = assessment_rules_r.extract(range(0,4),[0])
    assessment_rules_D_r = assessment_rules_r.extract(range(4, 8),[0])
    assessment_if_C_r = calculate_proba_assessment(assessment_rules_C_r)
    assessment_if_D_r = calculate_proba_assessment(assessment_rules_D_r)

    # Calcul of equation of resident on mutant p_mr
    freq_G_in_mr = assessment_if_C_r * proba_C_m + \
                   assessment_if_D_r * (1 - proba_C_m)
    if error == "assessment" or error == "both":
        freq_G_in_mr = freq_G_in_mr * (1-epsilon) + epsilon * (1-p_i)
    freq_G_in_mr = freq_G_in_mr.subs([(p_i, p_mr), (p_j, p_rr)])
    recursion_equa_p_mr = p_mr + q * (freq_G_in_mr - p_mr)
    difference_equa_p_mr = recursion_equa_p_mr - p_mr

    #We could solve this one directly
    freq_G_in_rm = assessment_if_C_m * proba_C_r + \
                   assessment_if_D_m * (1 - proba_C_r)
    if error == "assessment" or error == "both":
        freq_G_in_rm = freq_G_in_rm * (1-epsilon) + epsilon * (1-p_i)
    freq_G_in_rm = freq_G_in_rm.subs([(p_i, p_rm), (p_j, p_rr)])
    recursion_equa_p_rm = p_rm + q * (freq_G_in_rm - p_rm)
    difference_equa_p_rm = recursion_equa_p_rm - p_rm

    recursion_equa = [recursion_equa_p_mr,recursion_equa_p_rm]
    difference_equa = [difference_equa_p_mr,difference_equa_p_rm]

    analysis = dynamic_analysis(difference_equa, variables, [p_ini] * len(variables), 0, 1)

    #Fitness is the proportion of cooperation toward an individual
    fitness_m = benefit*proba_C_r.subs(p_j,analysis[p_mr]) - cost*proba_C_m.subs(p_j,analysis[p_rm])
    fitness_r = benefit*proba_C_r.subs(p_j, p_rr) - cost * proba_C_r.subs(p_j, p_rr)
    #Be careful with float error




    #res_equation = {"action_rules": "\"" +  "".join(str(elem) for elem in action_rules) + "\"",
    #                "assessment_rules": "\"" + "".join(str(elem) for elem in assessment_rules) + "\"",
    #                "action_rules_matrix": action_rules,
    #                "asssessment_rules_matrix": assessment_rules,
    #                "recursion_equa": recursion_equa,
    #                "difference_equa": difference_equa
    #}

    # Calcul of equation of mutant on resident p_rm

    res = {"action_rules_r": "\"" +  "".join(str(elem) for elem in action_rules_r) + "\"",
            "assessment_rules_r": "\"" +  "".join(str(elem) for elem in assessment_rules_r) + "\"",
            "rules_r": "\"" +  "".join(str(elem) for elem in action_rules_r) + "".join(str(elem) for elem in assessment_rules_r) + "\"",
            "action_rules_m": "\"" +  "".join(str(elem) for elem in action_rules_m) + "\"",
            "assessment_rules_m": "\"" +  "".join(str(elem) for elem in assessment_rules_m) + "\"",
            "rules_m": "\"" + "".join(str(elem) for elem in action_rules_m) + "".join(str(elem) for elem in assessment_rules_m) + "\"",
            "fitness_r":fitness_r, "fitness_m": fitness_m,"diff_fitness": fitness_m - fitness_r}
    #To do correlations
    res= {**res, **rules_matrix_to_list(action_rules_r,assessment_rules_r,"r"),**rules_matrix_to_list(action_rules_m,assessment_rules_m,"m")}

    return(res)




def sweep_ESS_analysis(input_file,benefit,cost):
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
                               donor_obs=parameters_input_file["donor_obs"], error = parameters_input_file["error"],
                               q = parameters_input_file["q"], epsilon = parameters_input_file["epsilon"],
                               p_ini = parameters_input_file["p_ini"],
                               benefit = benefit,cost = cost)
            list_res.append(res)
        print(index_r/len(dt_analytic.index))
    dt = pd.DataFrame.from_dict(list_res)
    name_file = "ESS-" + parameters_to_string(parameters_input_file, []) + "-ben=" + str(benefit) + "-cost=" + str(cost) + ".csv"
    dt.to_csv(name_file)








#===============================================SIMULATIONS==============================================================

#Detailed because we simulate explictly opinion (rather than reputation or just using the recursion equation)
#error is either "action" or "assessment"
def simulation_detailed(action_rules, assessment_rules, donor_obs, error,N, N_gen, q, epsilon, p_ini, N_print, plot, detail):
    #Initialisation--------------------------------------------------------
    #Initialise the matrix of reputation
    M_opinions = np.ones((N,N),dtype=int)
    for i in range(N):
        for j in range(N):
            if random.random() > p_ini:
                M_opinions[i,j] = 0
    #Initialise the table of results
    if donor_obs == True:
        p_by_gen = np.zeros((N_gen-N_print, N))
        p_ii_by_gen = np.zeros((N_gen-N_print,N))
    p_ik_by_gen = np.zeros((N_gen-N_print, N))
    cooperation_by_gen = np.zeros(N_gen-N_print)
    #Reshape the matrix as explained at the beginning of the code
    if donor_obs == True:
        action_rules_simul = np.reshape(np.flip(action_rules),(2,2))
    elif donor_obs == False:
        action_rules_simul = np.flip(action_rules)
    assessment_rules_simul = np.reshape(np.flip(assessment_rules),(2,2,2))
    #Start simulation-----------------------------------------------------
    for i in range(N_gen):
        #Choose randomly donor and recipient
        actors = np.random.choice(N, 2, replace=False)
        donor = actors[0]
        recipient = actors[1]
        #Choose donor action as a function of their action rules and the reputations of donor and recipient
        if donor_obs == True:
            donor_action = action_rules_simul[M_opinions[donor,donor],M_opinions[donor,recipient]]
        elif donor_obs == False:
            donor_action = action_rules_simul[M_opinions[donor,recipient]]
        if error == "action" or error == "both":
            if random.random() < epsilon:
                donor_action = 1-donor_action
        #Writing action
        if i >= N_print:
            cooperation_by_gen[i-N_print] = donor_action

        #Opinion update
        for j in range(N):
            #If donor don't update their opinions
            if donor_obs == False:
                if j == donor:
                    continue
            #Update opinion with some probability as a function of assessment rules and the action of the donor
            if random.random() < q:
                if error == "assessment" or error == "both":
                    if random.random() < epsilon:
                        M_opinions[j, donor] = 1 - M_opinions[j, donor]
                    else:
                        M_opinions[j,donor] = assessment_rules_simul[donor_action,M_opinions[j,donor],M_opinions[j,recipient]]
                else:
                    M_opinions[j, donor] = assessment_rules_simul[donor_action, M_opinions[j, donor], M_opinions[j, recipient]]

        #Write the reputations
        if i >= N_print:
            p_ik_by_gen[i-N_print, :] = M_opinions[:, :].mean(axis=0) - M_opinions[:, :].diagonal() / N
            if donor_obs == True:
                p_ii_by_gen[i-N_print,:] = M_opinions[:,:].diagonal().mean()
                p_by_gen[i-N_print,:] = M_opinions[:,:].mean(axis=0)

    #End of simulation------------------------------------------
    if plot == True:
        plt.plot(range(N_print,N_gen),p_ik_by_gen[:,0],range(N_print,N_gen),p_ik_by_gen[:,1],range(N_print,N_gen),p_ik_by_gen[:,2],range(N_print,N_gen),p_ik_by_gen[:,3],
                  p_ik_by_gen[:,:].mean())
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.show()
    #Level of detail determinate the output (either by generation or mean across reputation)
    if detail == 1:
        if donor_obs == True:
            res = {"p_ii_by_gen": p_ii_by_gen, "p_ik_by_gen": p_ik_by_gen, "p_by_gen": p_by_gen, "cooperation_by_gen": cooperation_by_gen}
        if donor_obs == False:
            res = {"p_ik_by_gen": p_ik_by_gen, "cooperation_by_gen": cooperation_by_gen}
    if detail == 0:
        if donor_obs == True:
            res = {"p_ii": p_ii_by_gen[:,:].mean(), "sd_p_ii": p_ii_by_gen[:,:].std(),
                   "p_ik": p_ik_by_gen[:,:].mean(), "sd_p_ik": p_ik_by_gen[:,:].std(),
                   "p": p_by_gen[:, :].mean(), "sd_p": p_by_gen[:, :].std(),
                   "cooperation": cooperation_by_gen[:].mean(),"std_cooperation": cooperation_by_gen[:].std()}
        if donor_obs == False:
            res = {"p_ik": p_ik_by_gen[:, :].mean(), "sd_p_ik": p_ik_by_gen[:, :].std(),
                   "cooperation": cooperation_by_gen[:].mean(),"std_cooperation": cooperation_by_gen[:].std()}
    return(res)


def simulations_detailed_replicated(action_rules, assessment_rules, donor_obs, error,N, N_gen, q, epsilon, p_ini, N_print, N_simul):
    list_res=[]
    for i in range(N_simul):
        res_single_run = simulation_detailed(action_rules, assessment_rules,
                                                          donor_obs, error,
                                                          N, N_gen, q, epsilon, p_ini,
                                                          N_print, False, 0)
        list_res.append(res_single_run)
    dt_by_simul=pd.DataFrame.from_dict(list_res)
    dt_mean= dt_by_simul.mean().add_prefix("mean_simul_")
    dt_se = dt_by_simul.sem().add_prefix("se_simul_")
    res = pd.concat([dt_mean,dt_se]).to_dict()
    return(res)

#Simulations for list of strategies
def sweep_simulations(input_file,  N_gen, N_print, N_simul, write):
    #Read a file with analytical result and get the parameters from it
    dt_analytic = pd.read_csv(input_file)
    parameters_input_file=read_parameters(input_file)
    list_res=[]
    #Simulations for each strategies
    for index, row in dt_analytic.iterrows():
        simulations = simulations_detailed_replicated(np.array([int(i) for i in row["action_rules"].replace("\"", "")]),
                                                      np.array([int(i) for i in row["assessment_rules"].replace("\"", "")]),
                                                      parameters_input_file["donor_obs"],parameters_input_file["error"],
                                                      parameters_input_file["N"], N_gen, parameters_input_file["q"], parameters_input_file["epsilon"], parameters_input_file["p_ini"],
                                                      N_print,N_simul)

        simulations.update({"diff_p_ik": np.abs(row["p_ik"] - simulations["mean_simul_p_ik"])})
        if parameters_input_file["donor_obs"] == True:
            simulations.update({"diff_p_ii": np.abs(row["p_ii"] - simulations["mean_simul_p_ii"])})
            simulations.update({"diff_p": np.abs(row["p"] - simulations["mean_simul_p"])})
        simulations.update({"diff_coop": np.abs(row["cooperation"] - simulations["mean_simul_cooperation"])})

        list_res.append(simulations)
        print(index / len(dt_analytic.index))
    dt_simul=pd.DataFrame.from_dict(list_res)
    dt_res = pd.concat((dt_analytic,dt_simul),axis=1)
    if write == True:
        name_file = "simul-" + parameters_to_string(parameters_input_file,[]) + "-gen=" + str(N_gen/1000) + "k" + "-print=" + str(N_print/1000) + "k" + "-S=" + str(N_simul) + ".csv"
        dt_res.to_csv(name_file)
    return(dt_res)


###==================================================ANALYSIS=======================================================================
#--------------------------------------Sandbox---------------------------------------------------------
#print(ESS_analysis(action_rules_r=sym.Matrix([1,0]),assessment_rules_r=sym.Matrix([1,1,1,1,0,1,1,0]),p_rr=0.5,
#                   action_rules_m=sym.Matrix([1,1]),assessment_rules_m=sym.Matrix([1,1,1,1,0,0,0,1]),
#             donor_obs=False,error="assessment",p_ini=0.5,epsilon=0.01,q=0.1,benefit=1,cost=0))


input_file = "analysis-sample=0-seed=1-donor_obs=False-error=action-q=0.05-epsilon=0.05-N=100-p_ini=0.5.csv"
#sweep_ESS_analysis(input_file = input_file,benefit=2,cost=1)

input_file = "analysis-sample=0-seed=1-donor_obs=False-error=assessment-q=0.05-epsilon=0.005-N=100-p_ini=0.5.csv"
#sweep_ESS_analysis(input_file = input_file,benefit=2,cost=1)
#sweep_ESS_analysis(input_file = input_file,benefit=5,cost=1)


# If same strategies

#----------------------------------For particular strategy------------------------------------------------

#print(analysis(sym.Matrix([1,1]),sym.Matrix([1,1,0,0,1,1,0,0]),False,"action",0.05,0.05,100,0.5))



#print(simulation_detailed(N=100,N_gen=100000,q=0.8,epsilon=0.8,p_ini=0.33,
#    action_rules = np.array([1,0]),assessment_rules=np.array([1,0,1,0,1,0,1,0]),donor_obs=False,plot=True,error="assessment",detail=0,N_print=00))

#----------------------To compare the results across different values of parameterrs-----------------------------------------

# for i in [0.01,0.1,0.5]:
#     for j in [0,0.01,0.1,]:
#         print("q =",i, end = ' ')
#         print("epsilon = ", j , end = ' ')
#         print(simulation_detailed(N=100,N_gen=800000,q=i,epsilon=j,p_ini=0.5,
#             action_rules = np.array([0,1,0,0]),assessment_rules=np.array([0,1,0,0,1,1,1,0]),donor_obs=True,plot=False,error="action",detail=0,N_print=400000)["p_ik"]-
#               analysis(sym.Matrix([0,1,0,0]),sym.Matrix([0,1,0,0,1,1,1,0]),True,"action",i,j,100,0.5)[p_ik])

#---------------------------------Analysis for list of strategies-----------------------------------------------


#sweep_rules_analysis(write=True,mirror=False,sample=100,seed=1,
#                   donor_obs=True, error="action",
#                   q=0.05,epsilon=0.05,N=100,p_ini=0.5)


#sweep_rules_analysis(write=True,mirror=False,sample=100,seed=1,
#                      donor_obs=True, error="assessment",
#                      q=0.05,epsilon=0.05,N=100,p_ini=0.5)

#sweep_rules_analysis(write=True,mirror=False,sample=100,seed=1,
#                   donor_obs=False, error="action",
#                    q=0.05,epsilon=0.05,N=100,p_ini=0.5)

#sweep_rules_analysis(write=True,mirror=False,sample=100,seed=1,
#                      donor_obs=False, error="assessment",
#                      q=0.05,epsilon=0.05,N=100,p_ini=0.5)

#sweep_rules_analysis(write=True,mirror=False,sample=100,seed=1,
#                      donor_obs=False, error="both",
#                      q=0.05,epsilon=0.05,N=100,p_ini=0.5)




#---------------------------------------For simulations across a list of strategies-----------------------------------------
start_time = time.time()
input_file = "analysis-sample=100-seed=1-donor_obs=True-error=action-q=0.05-epsilon=0.05-N=100-p_ini=0.5.csv"
#sweep_simulations(input_file=input_file, N_gen=250000, N_print=200000, N_simul=5 , write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=100-seed=1-donor_obs=True-error=assessment-q=0.05-epsilon=0.05-N=100-p_ini=0.5.csv"
#sweep_simulations(input_file=input_file, N_gen=250000, N_print=200000, N_simul=5 , write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=100-seed=1-donor_obs=False-error=action-q=0.05-epsilon=0.05-N=100-p_ini=0.5.csv"
#sweep_simulations(input_file=input_file, N_gen=250000, N_print=200000, N_simul=5 , write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=100-seed=1-donor_obs=False-error=assessment-q=0.05-epsilon=0.05-N=100-p_ini=0.5.csv"
#sweep_simulations(input_file=input_file, N_gen=250000, N_print=200000, N_simul=5 , write=True)
print("Execution time = " + str((time.time() - start_time)/60))
input_file = "analysis-sample=100-seed=1-donor_obs=False-error=both-q=0.05-epsilon=0.05-N=100-p_ini=0.5.csv"
sweep_simulations(input_file=input_file, N_gen=2500, N_print=2000, N_simul=5 , write=True)
print("Execution time = " + str((time.time() - start_time)/60))












#========================================ARCHIVES, TO UDPATE===============================================
#To print only equation and then calculate the value of stable points------------------------------------
def parser_string_to_math(input, is_list):
    #Recognise the list
    if input == "[]":
        output = []
        return(output)
    output = input.replace("[", "").replace("]", "").replace("\'", "").split(",")
    #Transform in mathematical expression
    output = list(map(sym.sympify,output))
    if is_list == False:
        output = output[0]
    return(output)

def analysis_equa(action_rules,assessment_rules,plot):
    proba_C = calculate_proba_C(action_rules)
    proba_D = calculate_proba_D(action_rules)
    freq_G_in_observers = calculate_freq_G_in_observers(assessment_rules,proba_C)
    recursion_equa = p_D + q*(freq_G_in_observers - p_D)
    recursion_equa_at_eq = recursion_equa.subs([(p_D, p_eq),(p_R,p_eq)])
    difference_equa_at_eq = recursion_equa_at_eq - p_eq
    derivative_recursion_equa_at_eq = sym.diff(recursion_equa_at_eq, p_eq)
    derivative_second_recursion_equa_at_eq = sym.diff(derivative_recursion_equa_at_eq, p_eq)

    eq_points = find_eq_points(difference_equa_at_eq,0,1)
    # stability analysis
    derivative_eq_points = []
    derivative_second_eq_points = []
    for i, e in enumerate(eq_points):
        derivative_eq_points.append(derivative_recursion_equa_at_eq.subs(p_eq, e))
        derivative_second_eq_points.append(derivative_second_recursion_equa_at_eq.subs(p_eq, e) * 0.5)


    if plot == True:
        #Plot recursion function
        lam_equa_prime_at_eq = sym.lambdify((p_eq), recursion_equa_at_eq.subs(q,0.1))
        plt.subplot(121)
        plt.title('Recursion equation')
        plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01),np.arange(0,1,0.01), np.vectorize(lam_equa_prime_at_eq)(np.arange(0,1,0.01)))

        #Plot difference equation (look at the difference with 0)
        plt.subplot(122)
        lam_equa_prime_at_eq = sym.lambdify((p_eq), recursion_equa_at_eq.subs(q,0.1) - p_eq)
        plt.title('Difference equation')
        plt.plot(np.arange(0, 1, 0.01),np.repeat(0,100),np.arange(0, 1, 0.01),np.vectorize(lam_equa_prime_at_eq)(np.arange(0, 1, 0.01)))

        #axes = plt.gca()
        #axes.set_ylim([0, 1])
        #axes.set_xlim([0, 1])
        #plt.savefig('dynamics_leading5.pdf')
        plt.show()


    res = {
        "action_rules": "\"" +  "".join(str(elem) for elem in action_rules) + "\"",
        "assessment_rules": "\"" + "".join(str(elem) for elem in assessment_rules) + "\"",
        "action_rules_matrix": action_rules,
        "asssessment_rules_matrix": assessment_rules,
        "recursion_equa": recursion_equa,
        "recursion_equa_at_eq": recursion_equa_at_eq,
        "derivative_recursion_equa_at_eq": derivative_recursion_equa_at_eq,
        "derivative_second_recursion_equa_at_eq": derivative_second_recursion_equa_at_eq,
        "equilibrium_points": eq_points,
        "derivative_at_equilibrium_points": list(map(sym.expand,derivative_eq_points)),
        "derivative_second_at_equilibrium_points": list(map(sym.expand,derivative_second_eq_points)),
    }
    return(res)


def sweep_rules_equa(write,mirror_image):
    counter = 0
    #We start with one so we remove mirror image that use 0 as good
    possible_action_rules = list(map(np.array, itertools.product([1, 0], repeat=4)))
    possible_assessment_rules = list(map(np.array, itertools.product([1, 0], repeat=8)))
    possible_rules = list(map(list,itertools.product(possible_action_rules,possible_assessment_rules)))
    #Choose the one that sounds good to us? Look at assessment on cooperators?
    if mirror_image == False:
        for i in possible_rules:
            for j, e in enumerate(possible_rules):
                if (e[0] == i[0][::-1]).all() and (e[1] == 1 - i[1]).all():
                    possible_rules.pop(j)
            counter += 1
            if counter%100 == 0:
                print("Removing mirror images : ", (counter / 2048))
    counter = 0
    list_res=[]
    for i in possible_rules:
            action_rules = sym.Matrix(i[0])
            assessment_rules = sym.Matrix(i[1])
            res = analysis(action_rules,assessment_rules,False)
            list_res.append(res)
            counter += 1
            if counter % 100 == 0:
                print("Computing equations: ",counter / (len(possible_rules)))
    dt = pd.DataFrame.from_dict(list_res)
    if write == True:
        dt.to_csv("table_equations_no_mirror.csv",)
    else:
        return(dt)

    def sweep_rules_equilibria(input_file, q_value, write, sample, sample_size, seed):
        dt_equations = pd.read_csv(input_file)
        if sample == True:
            dt_equations = dt_equations.sample(sample_size, random_state=seed)
        else:
            sample_size = 0
        total_counter = len(dt_equations.index)
        list_res = []
        counter = 0
        for index, row in dt_equations.iterrows():
            stable_points = []
            # Check if plateau
            plateau = 0
            recursion_equa_at_eq = parser_string_to_math(row["recursion_equa_at_eq"], is_list=False)
            lambda_difference_at_eq = sym.lambdify([p_eq, q], recursion_equa_at_eq - p_eq)
            eq_points = parser_string_to_math(row["equilibrium_points"], is_list=True)
            # If no equilibrium, stable points is one of the boundary or all (it is never one of the boundary in our case but it is more robust to check)
            if eq_points:
                derivative_at_eq_points = parser_string_to_math(row["derivative_at_equilibrium_points"], is_list=True)
                for i, e in enumerate(eq_points):
                    lambda_derivative_at_eq_points = sym.lambdify([q], derivative_at_eq_points[i])
                    if np.abs(lambda_derivative_at_eq_points(q_value)) < 1:
                        stable_points.append(e)
                    # If the derivative is 1, we look at the shape of the curve between the equilibrium points and the closest equilibrium point
                    if np.abs(lambda_derivative_at_eq_points(q_value)) == 1:
                        plateau = 1
                        # I think it happen only for boundaries so it is not necessary to check for other equilibrium, we do it to be sure
                        closest_equilibrium = 1 - e
                        # Find the closest equilibrium point
                        for e2 in eq_points:
                            if e2 == e:
                                continue
                            if np.abs(e - e2) < np.abs(e - closest_equilibrium):
                                closest_equilibrium = e2
                        if lambda_difference_at_eq(min(e, closest_equilibrium) + np.abs((e - closest_equilibrium) / 2),
                                                   q_value) > 0 and e > closest_equilibrium:
                            stable_points.append(e)
                        if lambda_difference_at_eq(min(e, closest_equilibrium) + np.abs((e - closest_equilibrium) / 2),
                                                   q_value) < 0 and e < closest_equilibrium:
                            stable_points.append(e)
            if not stable_points:
                if lambda_difference_at_eq(0.5, q_value) > 0:
                    stable_points.append(1)
                if lambda_difference_at_eq(0.5, q_value) < 0:
                    stable_points.append(0)
            # Calculate predicted_eq and fitness
            action_rules_matrix = sym.sympify(row["action_rules_matrix"])
            proba_C = calculate_proba_C(action_rules_matrix)
            fitness = 0
            if not stable_points:
                predicted_p_eq = 0.5
                fitness = proba_C.subs([(p_D, 0.5), (p_R, 0.5)])
            if len(stable_points) == 1:
                predicted_p_eq = stable_points[0]
                fitness = proba_C.subs([(p_D, stable_points[0]), (p_R, stable_points[0])])
            if len(stable_points) == 2:
                # The average weigthed by the bassin of attraction delimited by unstable point
                unstable_point = next(iter((set(eq_points).difference(set(stable_points)))))
                predicted_p_eq = np.abs(stable_points[0] - unstable_point) * stable_points[0] + (
                            1 - np.abs(stable_points[0] - unstable_point)) * stable_points[1]
                fitness = np.abs(stable_points[0] - unstable_point) * proba_C.subs(
                    [(p_D, stable_points[0]), (p_R, stable_points[0])]) + (
                                      1 - np.abs(stable_points[0] - unstable_point)) * proba_C.subs(
                    [(p_D, stable_points[1]), (p_R, stable_points[1])])

            res = {
                "action_rules": row["action_rules"],
                "assessment_rules": row["assessment_rules"],
                "N_stable_points": len(stable_points),
                "predicted_p_eq": predicted_p_eq,
                "fitness": fitness,
                "plateau": plateau
            }
            list_res.append(res)
            print(counter / total_counter)
            counter += 1
        dt = pd.DataFrame.from_dict(list_res)
        if write == True:
            name_file = "stable_fitness" + "-file=" + str(input_file).replace(".csv", "") + "-q=" + str(q_value)
            if sample == True:
                name_file += "-sample_size=" + str(sample_size) + "-seed=" + str(seed)
            name_file += ".csv"
            dt.to_csv(name_file)
        else:
            return (dt)






#---------------------------------------------Mixed population---------------------------------------------------------
#To update
def analysis_mixed(leading_strat,other_strat,f_L,f_noL):
    proba_C = calculate_proba_C(leading_strat.action_rules)
    proba_D = calculate_proba_D(leading_strat.action_rules)
    freq_G_in_observers = calculate_freq_G_in_observers(leading_strat.assessment_rules, proba_C, proba_D)
    freq_G_in_observers_L_L = freq_G_in_observers.subs([(p_i, p_eq_L),(p_j,p_eq_L)])
    freq_G_in_observers_L_noL = freq_G_in_observers.subs([(p_i, p_eq_L), (p_j, p_eq_noL)])
    p_L_recursion_equa = p_eq_L + f_L*q*(f_L*freq_G_in_observers_L_L + f_noL*freq_G_in_observers_L_noL - p_eq_L)
    p_L_difference_equa = p_L_recursion_equa - p_eq_L
    print("recursion L",p_L_recursion_equa)
    print("difference L",p_L_difference_equa)

    proba_C = calculate_proba_C(AllD.action_rules)
    proba_D = calculate_proba_D(AllD.action_rules)
    freq_G_in_observers = calculate_freq_G_in_observers(leading_strat.assessment_rules, proba_C, proba_D)
    #freq_G_in_observers_noL = freq_G_in_observers.subs([(p_D, p_eq_noL),(p_R,p_eq_noL)])
    #We could just calculate freG and replace p_R by f_L*p_L + f_noL * p_noL?
    freq_G_in_observers_noL_L = freq_G_in_observers.subs([(p_i, p_eq_noL),(p_j,p_eq_L)])
    freq_G_in_observers_noL_noL = freq_G_in_observers.subs([(p_i, p_eq_noL), (p_j, p_eq_noL)])
    #No need for (1-f_L)*q * p_eq_noL because they are not taken in account here. Low number of L means big change
    p_noL_recursion_equa = p_eq_noL + f_L* q * (f_L*freq_G_in_observers_noL_L + f_noL*freq_G_in_observers_noL_noL - p_eq_noL)
    p_noL_difference_equa = p_noL_recursion_equa - p_eq_noL
    print("recursion noL",p_noL_recursion_equa)
    print("difference noL",p_noL_difference_equa)


    system_equa = [p_L_difference_equa/q,p_noL_difference_equa/q]
    print(system_equa)
    eq_points=sym.solve(system_equa,set=True)
    print(eq_points)

    matrix_recursion_equas = sym.Matrix([p_L_recursion_equa,p_noL_recursion_equa])
    matrix_variable = sym.Matrix([p_eq_L,p_eq_noL])
    jacobian_matrix= matrix_recursion_equas.jacobian(matrix_variable)

    for i in eq_points[1]:
        print("0",i)
        jacobian_matrix_at_eq=jacobian_matrix.subs([(p_eq_L, i[0]),(p_eq_noL,i[1])])
        eigenvalues=jacobian_matrix_at_eq.eigenvals(multiple=True)
        print("equilibrium points", i)
        for j in eigenvalues:
            print(sym.sympify(j).subs([(q,0.1)]).evalf())

    #p_D_recursion_equa = p_D + q * (freq_G_in_observers - p_D)
    #p_D_recursion_equa_at_eq = p_D_recursion_equa.subs([(p_D, p_L_eq), (p_R, p_L_eq)])
    #difference_equa_at_eq = recursion_equa_at_eq - p_L_eq


#analysis_mixed(Leading_five,AllD,0.5,0.5)



def simulation_mixed_detailed(N,N_Leading,N_gen,q,epsilon,p_t0,strat_leading,strat_not_leading,plot):
    #Initialise the matrix of reputation
    M_opinions = np.ones((2,N,N),dtype=int)
    reputations = np.zeros((N_gen,N))
    for i in range(N):
        for j in range(N):
            if random.random() < p_t0:
                M_opinions[0,i,j] = 0

    #Initialise the population
    pop = []
    for i in range(N_Leading):
        pop.append(strat_leading())
    for i in range(N_Leading,N):
        pop.append(strat_not_leading())
    #Start simulation
    for i in range(N_gen):
        #Copy the previous reputation of everyone
        M_opinions[1,:,:] = M_opinions[0,:,:]
        #Choose donor and recipient
        actors = np.random.choice(N, 2, replace=False)
        donor = pop[actors[0]]
        recipient = pop[actors[1]]
        #Choose donor action as a function of their action rules and the reputations of donor and recipient
        donor.action = donor.action_rules_simul[M_opinions[0,actors[0],actors[0]],M_opinions[0,actors[0],actors[1]]]
        #Do mistake in their action with some probability
        if random.random() < epsilon:
            if donor.action == 0:
                donor.action = 1
            else:
                donor.action = 0
        #Update opinion (only leading individuals have an opinion)
        for j in range(N_Leading):
            #Recipient and donor don't update their opinions
            #Recipient and donor don't update their opinions
            if j == actors[0] or j == actors[1]:
                continue
            #Update opinion with some probability as a function of assessment rules and the action of the donor
            if random.random() < q:
                M_opinions[1,j,actors[0]] = pop[j].assessment_rules_simul[donor.action,M_opinions[1,j,actors[0]],M_opinions[1,j,actors[1]]]

        M_opinions[0, :, :] = M_opinions[1, :, :]
        reputations[i,:] = M_opinions[0,:,:].mean(axis=1)

    #End of simulation


    #Plots-----------------------------------
    if plot == True:
        f = plt.figure(figsize=(15, 8))

        #Plot of mean value of reputation, mean reputation for leadings, mean reputation for not leadings
        f.add_subplot(131)
        plt.plot(range(N_gen+1), np.mean(M_opinions[:, 0:N_Leading, :], axis=(1, 2)),range(N_gen+1), np.mean(M_opinions[:, 0:N_Leading, 0:N_Leading], axis=(1, 2)),range(N_gen+1),np.mean(M_opinions[:, 0:N_Leading, N_Leading:N], axis=(1, 2)))
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.grid()
        plt.legend(["mean","mean_leading","mean_notleading"],loc="upper left")

        # Plot of reputation of the first five individuals
        f.add_subplot(132)
        plt.plot(range(N_gen+1), np.mean(M_opinions[:, 0, :], axis=(1)),range(N_gen+1),  np.mean(M_opinions[:, 1, :], axis=(1)),range(N_gen+1),np.mean(M_opinions[:, 2, :], axis=(1)),range(N_gen+1),np.mean(M_opinions[:, 3, :], axis=(1)),range(N_gen+1),np.mean(M_opinions[:, 4, :], axis=(1)))
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.grid()

        # Plot of std value of reputation, std reputation for leadings, std reputation for not leadings
        f.add_subplot(133)
        plt.plot(range(N_gen+1), np.std(M_opinions[:, 0:N_Leading, :], axis=(1, 2)),range(N_gen+1), np.std(M_opinions[:, 0:N_Leading, 0:N_Leading], axis=(1, 2)),range(N_gen+1),np.std(M_opinions[:, 0:N_Leading, N_Leading:N], axis=(1, 2)))
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.grid()
        plt.legend(["std", "std_leading", "std_notleading"],loc="upper left")

        #Name file with parameters values
        parameters = ""
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for i in args:
            #Parameters to not print
            if i == "strat_leading" or i == "strat_not_leading":
                continue
            parameters = parameters + "-" + i
            parameters = parameters + "=" + str(values[i])
        print(parameters)
        name_file= "E:/OneDrive - Teeside University/OneDrive - Teesside University/Research/A1-Projects/2020_RepDyn/Res/" + "RepDyn_simul-" + str(strat_leading.__name__) + str(parameters) +".pdf"
        plt.savefig(name_file)

    return(reputations)


# To look at
# p_D_prime_pertubation = p_D_prime.subs([(p_D, p_R + x),(p_R,p_R)])
# print(p_D_prime_pertubation)
# p_D_prime_pertubation = p_D_prime_pertubation.subs(p_R,0.8)
# print("uihiujh",p_D_prime_pertubation)
# lam = sym.lambdify((x),p_D_prime_pertubation)
# plt.subplot(133)
# plt.plot(np.arange(-1,1,0.01,), np.vectorize(lam)(np.arange(-1,1,0.01)))
# plt.grid()
# axes = plt.gca()
# axes.set_ylim([-1, 1])
# axes.set_xlim([-1, 1])


##-----------------------------Simulations with the recursion equation-----------------------------------
# proba_C = calculate_proba_C(Leading_five.action_rules)
# proba_D = calculate_proba_D(Leading_five.action_rules)
# p_D_prime = calculate_freq_G_in_observers(Leading_five.assessment_rules, proba_C, proba_D)
# lam_p_D_prime = sym.lambdify((p_D,p_R),p_D_prime)
# N = 50
# N_gen = 1000
# N_O = 1
# #pop = np.random.rand(N_gen+1,N)
# pop = np.ones((N_gen+1,N))
# pop[0,:]=np.repeat(0.1,N)
# #pop[0,0]=0.5
#
# print(lam_p_D_prime(1,1))
# for i in range(0,N_gen):
#     #First is donor, Second is recipient, Third is observer (no need for observer for now)
#     actors = np.random.choice(N,3,replace=False)
#     pop[i+1,:]=pop[i,:]
#     pop[i+1,actors[0]] = pop[i,actors[0]] + lam_p_D_prime(pop[i,actors[0]],pop[i,actors[1]])

#Update all donors -> Does not change much
# for i in range(0,N_gen):
#     pop[i+1,:]=pop[i,:]
#     for j in range (0,N):
#         actors = np.random.choice(list(range(0,j)) + list(range(j,N)), 1, replace=False)
#         pop[i+1,j] = pop[i,j] + N_O/N * lam_p_D_prime(pop[i,j],pop[i,actors[0]])

#As a function of the mean -> The same
# for i in range(0,N_gen):
#     pop[i+1,:]=pop[i,:]
#     actors = np.random.choice(N,1,replace=False)
#     pop[i+1,actors[0]] = ((N-N_O)/N) * pop[i,actors[0]] + N_O/N * lam_p_D_prime(pop[i,actors[0]],np.mean(pop[i,:]))

#As a function of itself
# print((N_O/N))
# for i in range(0,N_gen):
#     pop[i+1,:]=pop[i,:]
#     actors = np.random.choice(N,1,replace=False)
#     pop[i+1,actors[0]] =  pop[i,actors[0]] + (N_O/N) * lam_p_D_prime(pop[i,actors[0]],pop[i,actors[0]])

# plt.plot(range(N_gen+1),np.mean(pop,axis=1),pop[:, 1],pop[:, 2],pop[:, 3],pop[:, 4])
# axes = plt.gca()
# axes.set_ylim([0, 1])
# plt.show()
