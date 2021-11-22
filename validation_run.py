# This code is used to generate simulations using parameters from the single outbreaks. Each set of parameters will generate 5000 valid results (We only consider the simulations with number of introductions on day 1 > 0)
# Please note that the code below requires significant amount of computational resources since in order to generate valid results, the number of total simulations can reach to 10E13.

## import libraries
import numpy as np
import pandas as pd

from collections import namedtuple
from itertools import product
from scipy.stats import binom
from scipy.stats.mstats import gmean
from scipy.interpolate import Rbf

from sklearn.preprocessing import power_transform
import math
import pickle
import itertools

import importlib
import statsmodels.api as st
import sys

## import code
import covid_workplace_model as new_model

print("version: ", new_model.version())

######## Parameter input ##########
######## Please comment out unused parameters ############

# ## Tianjin office
# print("*"*5, 'Tianjin office', '*'*5)
# EMPLOYEE=906
# WFH=0
# CASE_RATE=1/100000
# DAILY_AVERAGE_CONTACTS=[4,10]
# DESIGN_ACH=[3,8]
# SPEAKING_PCT=[.05,.25]
# NUMBER_DAYS=10
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=7

## Korean call center
print("*" * 5, "Korean Call Center", "*" * 5)
EMPLOYEE = 216
WFH = 0
CASE_RATE = 1 / 1e5
DAILY_AVERAGE_CONTACTS = np.asarray([4, 10])
DESIGN_ACH = [3, 8]
SPEAKING_PCT = np.asarray([0.55, 0.66])
SPEAKING_VOL = 70
NUMBER_DAYS = 14
PROBABILITY_CONTACT_REMAIN = 0.4
EMPIRICAL = 76

# ## San Diego VA Office #
# print("*"*5, 'San Diego VA office', '*'*5)
# EMPLOYEE=100
# WFH=0
# CASE_RATE=1/100000
# DAILY_AVERAGE_CONTACTS=[4,10]
# DESIGN_ACH=[3,8]
# SPEAKING_PCT=[.05,.4]
# NUMBER_DAYS=8
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=5

# ## singapore conference
# print("*"*5, 'Singapore conference', '*'*5)
# EMPLOYEE=111
# WFH=0
# CASE_RATE=1/1e5
# DAILY_AVERAGE_CONTACTS=[6,12]
# DESIGN_ACH=[3,8]
# SPEAKING_PCT=[.25,.4]
# NUMBER_DAYS=3
# PROBABILITY_CONTACT_REMAIN=0
# EMPIRICAL=7

# ## SD meat processing Shift 1
# print("*"*5, 'SD Meat Processing Shfit 1', '*'*5)
# EMPLOYEE=1744
# WFH=0
# CASE_RATE=1/1e5
# DAILY_AVERAGE_CONTACTS=[4,10]
# DESIGN_ACH=[2,22]
# SPEAKING_PCT=[.05,.2]
# NUMBER_DAYS=14
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=32

# ## SD meat processing shift 2
# print("*"*5, 'SD Meat Processing Shfit 2', '*'*5)
# EMPLOYEE=1459
# WFH=0
# CASE_RATE=1/1e5
# DAILY_AVERAGE_CONTACTS=[4,10]
# DESIGN_ACH=[2,22]
# SPEAKING_PCT=[.05,.2]
# NUMBER_DAYS=14
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=6

# ## Henan expressway
# print("*"*5, 'Henan Expressway', '*'*5)
# ## to run this case, skip the block below to calculate SAR,  instead enter SAR = [0.2, 0.2]
# EMPLOYEE=103
# WFH=0
# CASE_RATE=1/100000
# DAILY_AVERAGE_CONTACTS=[2,4]
# DESIGN_ACH=[3,8]
# SPEAKING_PCT=[.05,.25]
# NUMBER_DAYS=13
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=6

# ## MLB with mask protocol and social distancing
# print("*"*5, 'MLB with mask and social distancing', '*'*5)
# EMPLOYEE=68
# WFH=0
# CASE_RATE=60/1e5
# DAILY_AVERAGE_CONTACTS=[4,8]
# DESIGN_ACH=[3,8]
# SPEAKING_PCT=[.05,.25]
# NUMBER_DAYS=10
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=20

# ## MLB without mask protocol and social distancing
# print("*"*5, 'MLB without mask and social distancing', '*'*5)
# EMPLOYEE=68
# WFH=0
# CASE_RATE=60/1e5
# DAILY_AVERAGE_CONTACTS=[4,10]
# DESIGN_ACH=[3,8]
# SPEAKING_PCT=[.05,.25]
# NUMBER_DAYS=10
# PROBABILITY_CONTACT_REMAIN=.4
# EMPIRICAL=20

######## simulations ##########
######## Please note that this takes significant amount of time ############

## calculate SAR
## skip this for Henan Expressway Outbreak
## Henan Expressway SAR = np.asarray([0.2, 0.2])
SAR = np.asarray([])
for pct in SPEAKING_PCT:
    SAR = np.append(
        SAR,
        new_model.get_sar_from_speaking_airflow(
            design_airflow_ACH=DESIGN_ACH,
            filtration_per_hour=0,
            speaking_percentage=pct,
            speaking_volume=SPEAKING_VOL,
            mask_effectiveness=0.0,
        ),
    )

## calculate the number of simulation required
## this follows negative binomial distribution
VALID_SIM = 5000
df = np.random.negative_binomial(VALID_SIM / CASE_RATE, CASE_RATE, VALID_SIM)
N_SIM = np.ceiling(np.mean(df))

## generate simulation based on four sets of parameters
sim = 0
res = np.zeros((N_SIM * 4, NUMBER_DAYS))
for sar in np.asarray([np.min(SAR), np.max(SAR)]):
    for contact_size in DAILY_AVERAGE_CONTACTS:
        res[sim : (sim + N_SIM)][:] = new_model.calculate_nday_workplace_incidence(
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
