# COVID-19 model for Workplace Outbreaks
> Guo, X.; Gupta, A.; Sampat, A; Wei, C. A Stochastic Contact Network Model for Assessing Outbreak Risk of COVID-19 in Workplaces.

A simulation-based stochastic contact network to assess the cumulative incidence of COVID-19 in workplaces. 

## Getting started

This code is written in `python:3`, and is expected to work in a standard python environment with the following libraries.

### Dependencies
```
numpy
scipy
```

## Quick Usage

In a terminal, type the following command to save 10,000 simulations of cumulative incidence in a workplace after 10 days in a csv file.


```
python covid_workplace_model.py -e 1000 -c 0.001 -o 'workplace_outbreak.csv'
```

The required arguments are `-e`, `-c`, and `-o` for number of employees, daily new case rate, and output file location, respectively.

### Configuration

The following arguments are accepted - 

#### -e | --employees
Required: `True`  
Type: `Integer`  

#### -c | --caserate
Required: `True`  
Type: `Float [0., 1.]`  

#### -o | --outfile
Required: `True`  
Type: `String filepath`  

#### --contacts
Required: `False`  
Type: `Integer`  
Default: 6

#### --sar
Required: `False`  
Type: `Float [0., 1.]`  
Default: 0.051

#### --days
Required: `False`  
Type: `Integer`  
Default: 10

#### --sims
Required: `False`  
Type: `Integer`  
Default: 10000

#### --prob_sustain
Required: `False`  
Type: `Float [0., 1.]`  
Default: 0.6

#### -v | --verbosity
Required: `False`  
Type: `Integer {0, 1, 2}`  
Default: 0

#### -l | --logfile
Required: `False`  
Type: `String filepath`  
Default : `None`  


## Advanced Usage

The python file provides functions for other equations from the paper. These functions can be used by importing them in an ipython terminal or another python project. 

```
from covid_workplace_model import *
```

The following functions are provided - 
* `get_activity_multiplier` - Returns speaking activity multiplier for a given speaking volume in dB.
* `get_activity_virion` - Returns the activity virions based on speaking percentage and activity_multiplier
* `get_sar_from_speaking_airflow` - Returns the SAR values, given design airflow in ACH, filteration, speaking percentage, speaking volume, and mask effectiveness ratio
* `get_betadist_parameters` - Returns alpha and beta parameters for a Beta distribution given the average SAR
* `get_sar_given_betadist_parameters` - Returns average SAR values, given alpha and beta parameters for a Beta distribution
* `get_probability_introduction_each_day` - Returns the probability of case introduction based on employees, and daily new case rate
* `calculate_nday_workplace_incidence` - Returns the cumulative incidence in a workplace over N days

## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

## Licensing

The code in this project is licensed under GNU General Public License v3.0.