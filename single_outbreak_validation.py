"""
Created by : Abhineet Gupta and Xi Guo
Created on : 2021-11-18
Last Modified by : Xi Guo
Last Modified on : 2021-12-03
Accompanying Paper : Guo, X.; Gupta, A.; Sampat, A; Wei, C. A Stochastic Contact Network Model for 
                        Assessing Outbreak Risk of COVID-19 in Workplaces.
"""

# This code is used to generate simulations using parameters from the single outbreaks. Each set of parameters will generate 5000 valid results (We only consider the simulations with number of introductions on day 1 > 0)
# Please note that the code below requires significant amount of computational resources since in order to generate valid results, the number of total simulations can reach to 10E13.

## import libraries
import numpy as np

## import code
import covid_workplace_model as covid_model

######## Parameter input ##########
VALID_SIM = 10
######## Please comment out unused parameters ############

## Tianjin office
EMPLOYEE = 906
WFH = 0
CASE_RATE = 1 / 100000
DAILY_AVERAGE_CONTACTS = [4, 10]
DESIGN_ACH = [3, 8]
SPEAKING_PCT = [0.05, 0.25]
NUMBER_DAYS = 10
PROBABILITY_CONTACT_REMAIN = 0.4
EMPIRICAL = 7

# ## Korean call center
# EMPLOYEE = 216
# WFH = 0
# CASE_RATE = 1 / 1e5
# DAILY_AVERAGE_CONTACTS = np.asarray([4, 10])
# DESIGN_ACH = [3, 8]
# SPEAKING_PCT = np.asarray([0.55, 0.66])
# SPEAKING_VOL = 70
# NUMBER_DAYS = 14
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 76

# ## San Diego VA Office #
# EMPLOYEE = 100
# WFH = 0
# CASE_RATE = 1 / 100000
# DAILY_AVERAGE_CONTACTS = [4, 10]
# DESIGN_ACH = [3, 8]
# SPEAKING_PCT = [0.05, 0.4]
# NUMBER_DAYS = 8
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 5

# ## singapore conference
# EMPLOYEE = 111
# WFH = 0
# CASE_RATE = 1 / 1e5
# DAILY_AVERAGE_CONTACTS = [6, 12]
# DESIGN_ACH = [3, 8]
# SPEAKING_PCT = [0.25, 0.4]
# NUMBER_DAYS = 3
# PROBABILITY_CONTACT_REMAIN = 0
# EMPIRICAL = 7

# ## SD meat processing Shift 1
# EMPLOYEE = 1744
# WFH = 0
# CASE_RATE = 1 / 1e5
# DAILY_AVERAGE_CONTACTS = [4, 10]
# DESIGN_ACH = [2, 22]
# SPEAKING_PCT = [0.05, 0.2]
# NUMBER_DAYS = 14
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 32

# ## SD meat processing shift 2
# EMPLOYEE = 1459
# WFH = 0
# CASE_RATE = 1 / 1e5
# DAILY_AVERAGE_CONTACTS = [4, 10]
# DESIGN_ACH = [2, 22]
# SPEAKING_PCT = [0.05, 0.2]
# NUMBER_DAYS = 14
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 6

# ## Henan expressway
# ## to run this case, skip the block below to calculate SAR,  instead enter SAR = [0.2, 0.2]
# EMPLOYEE = 103
# WFH = 0
# CASE_RATE = 1 / 100000
# DAILY_AVERAGE_CONTACTS = [2, 4]
# DESIGN_ACH = [3, 8]
# SPEAKING_PCT = [0.05, 0.25]
# NUMBER_DAYS = 13
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 6

# ## MLB with mask protocol and social distancing
# EMPLOYEE = 68
# WFH = 0
# CASE_RATE = 60 / 1e5
# DAILY_AVERAGE_CONTACTS = [4, 8]
# DESIGN_ACH = [3, 8]
# SPEAKING_PCT = [0.05, 0.25]
# NUMBER_DAYS = 10
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 20

# ## MLB without mask protocol and social distancing
# EMPLOYEE = 68
# WFH = 0
# CASE_RATE = 60 / 1e5
# DAILY_AVERAGE_CONTACTS = [4, 10]
# DESIGN_ACH = [3, 8]
# SPEAKING_PCT = [0.05, 0.25]
# NUMBER_DAYS = 10
# PROBABILITY_CONTACT_REMAIN = 0.4
# EMPIRICAL = 20

######## simulations ##########
######## Please note that this takes significant amount of time ############

## calculate SAR
## skip this for Henan Expressway Outbreak
## Henan Expressway sar = np.asarray([0.2, 0.2])
sar = np.asarray([])
for pct in SPEAKING_PCT:
    sar = np.append(
        sar,
        covid_model.get_sar_from_speaking_airflow(
            design_airflow_ACH=DESIGN_ACH,
            filtration_per_hour=0,
            speaking_percentage=pct,
            speaking_volume=SPEAKING_VOL,
            mask_effectiveness=0.0,
        ),
    )

## calculate the number of simulation required
## this follows negative binomial distribution
prob_day1_intro = covid_model.get_probability_introduction_each_day(
    employees_at_work=EMPLOYEE, case_rate=CASE_RATE
)
N_SIM = int(np.ceil(VALID_SIM / prob_day1_intro))
## generate simulation based on four sets of parameters
sim = 0
res = np.zeros((N_SIM * 4, NUMBER_DAYS))
for sar in np.asarray([np.min(sar), np.max(sar)]):
    for contact_size in DAILY_AVERAGE_CONTACTS:
        res[sim : (sim + N_SIM)][:] = covid_model.calculate_nday_workplace_incidence(
            employees_at_work=EMPLOYEE,
            case_rate=CASE_RATE,
            daily_contact_size=contact_size,
            probability_contacts_sustain=1 - PROBABILITY_CONTACT_REMAIN,
            sar_average=sar,
            num_days=NUMBER_DAYS,
            num_sims=N_SIM,
        )
    sim += N_SIM

print("Run completed")
