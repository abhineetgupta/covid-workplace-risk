"""
Created by : Abhineet Gupta and Xi Guo
Created on : 2021-04-21
Last Modified by : Abhineet Gupta
Last Modified on : 2021-04-
Accompanying Paper : Guo, X.; Gupta, A.; Sampat, A; Wei, C. A Stochastic Contact Network Model for 
                        Assessing Outbreak Risk of COVID-19 in Workplaces.
"""

import logging
import numpy as np
from scipy.stats import binom


def version():
    return "1.0.0"


def get_activity_multiplier(speaking_volume):
    """Returns speaking activity multiplier for a given speaking volume in dB.
    Calculation is based on Asadi, Sima, Anthony S. Wexler, Christopher D. Cappa, Santiago
    Barreda, Nicole M. Bouvier, and William D. Ristenpart. 2019. “Aerosol Emission and
    Superemission during Human Speech Increase with Voice Loudness.”
    Scientific Reports 9 (1): 2348. https://doi.org/10.1038/s41598-019-38808-z.
    -----Parameters-----
            speaking_volume : scalar or array of speaking volume in dB.
    -----Returns-----
            activity_multiplier : Multiplier for speaking volume given a reference volume
    """

    volume_reference = 60.0

    activity_multiplier_reference = 53.0 / 6.0
    volume_difference_reference = 98.0 - 70.0

    return np.power(
        activity_multiplier_reference,
        (speaking_volume - volume_reference) / volume_difference_reference,
    )


def get_activity_virion(speaking_percentage, speaking_volume):
    """Returns the activity virions based on speaking percentage and volume
    -----Parameters-----
            speaking_percentage: scalar or array of percentage time spent talking,
                yelling, singing, etc. in range [0., 1.]
            speaking_volume : scalar or array of speaking volume in dB.
    -----Returns-----
            activity_virions : Number of virions emitted
    """
    # Ratio of breathing to speaking varion based on paper used for SAR calculation
    speaking_virion = 46.0

    return np.asarray(
        1.0
        - speaking_percentage
        + (
            speaking_percentage
            * get_activity_multiplier(speaking_volume)
            * speaking_virion
        ),
        dtype=np.float64,
    )


def get_sar_from_speaking_airflow(
    design_airflow_ACH=4,
    filtration_per_hour=0,
    speaking_percentage=0.25,
    speaking_volume=65,
    mask_effectiveness=0.0,
):
    """Returns the SAR values, given design airflow in ACH, filteration, speaking percentage,
    speaking volume, and mask effectiveness ratio
    Based on Prentiss, Mara, Arthur Chu, and Karl K. Berggren. 2020. “Superspreading
    Events Without Superspreaders: Using High Attack Rate Events to Estimate Nº for
    Airborne Transmission of COVID-19.” MedRxiv, October, 2020.10.21.20216895.
    https://doi.org/10.1101/2020.10.21.20216895.
    -----Parameters-----
            design_airflow_ACH : scalar or array of design airflow in ACH. Actual airflow is half of this.
            filtration_per_hour : scalar or array of number of filter cycles per hour
            speaking_percentage: scalar or array of percentage time spent talking,
                yelling, singing, etc. in range [0., 1.]
            speaking volume : scalar or array of speaking volume in dB that can be
                used to estimate activity multiplier. Deafults to 60dB
            mask_effectiveness : scalar or array of mask effectiveness values in range [0., 1.]
    -----Returns-----
            sar : SAR values in the range [0, 1.]
    """
    virion_additional_decay_factor = 0.62

    speaking_percentage_benchmark_sar = 0.25
    speaking_volume_benchmark_sar = 65
    actual_airflow_benchmark_sar = 2.0
    benchmark_sar = 5.1 / 100

    inhaled_virions_benchmark_sar = -np.log(1.0 - benchmark_sar)

    # actual airflow is assumed to be half of design airflow.
    actual_airflow_ACH = np.asarray(design_airflow_ACH, dtype=np.float64) / 2.0

    inhaled_virions = (
        inhaled_virions_benchmark_sar
        * get_activity_virion(
            speaking_percentage=speaking_percentage,
            speaking_volume=speaking_volume,
        )
        / get_activity_virion(
            speaking_percentage=speaking_percentage_benchmark_sar,
            speaking_volume=speaking_volume_benchmark_sar,
        )
        * (actual_airflow_benchmark_sar + virion_additional_decay_factor)
        / (actual_airflow_ACH + virion_additional_decay_factor + filtration_per_hour)
    )
    sar = 1.0 - np.exp(-inhaled_virions * (1.0 - mask_effectiveness))
    return sar


def get_betadist_parameters(sar):
    """Returns alpha and beta parameters for a Beta distribution given the average SAR
    -----Parameters-----
            sar : array of SAR values in range [0., 1.]
    -----Returns-----
            (betadist_alpha, betadist_beta) for the main function to estimate infections
    """
    sar = np.asarray(sar, dtype=np.float64)
    betadist_alpha = 1.0
    if sar.size > 1:
        betadist_alpha = np.full_like(
            sar, betadist_alpha
        )  # assumed to be fixed right now
    betadist_beta = (1.0 - sar) * betadist_alpha / sar
    return (betadist_alpha, betadist_beta)


def get_sar_given_betadist_parameters(betadist_alpha, betadist_beta):
    """Returns average SAR values, given alpha and beta parameters for a Beta distribution
    -----Parameters-----
            betadist_alpha : scalar or array of alpha parameter for Beta distribution
            betadist_beta : scalar or array of beta parameter for Beta distribution
    -----Returns-----
            average SAR as the average of the Beta distribution
    """
    return betadist_alpha / (betadist_alpha + betadist_beta)


def get_probability_introduction_each_day(employees_at_work, case_rate):
    """Returns the probability of case introduction based on employees, and daily new case rate
    -----Parameters-----
            employees_at_work : pre covid population * (1 - working from home)
            case_rate : x-day running average for new cases per unit of population,
                        common ranges are 0.00001 to 0.001
    -----Returns-----
            probability of at least one case introduction in a day
    """
    return binom.sf(0, int(employees_at_work), case_rate)


def calculate_nday_workplace_incidence(
    employees_at_work,
    case_rate,
    daily_contact_size=6,
    probability_contacts_sustain=0.6,
    sar_average=5.1 / 100,
    num_days=10,
    num_sims=10000,
):
    """Returns the cumulative incidence in a workplace over N days (default: 10 days)
    -----Parameters-----
            employees_at_work : pre covid population * (1 - working from home)
            daily_contact_size : average number of daily contacts per employee
            probability_contacts_sustain : probability that contacts on consecutive days remain
                                            the same as the previous day. 1 implies that
                                            all contacts stay the same on consecutive days,
                                            while 0 implies that previous day's contacts have
                                            no consequence on next day's contacts.
            sar_average : average secondary attack rate in range [0, 1]
            case_rate : daily new cases per population, common ranges are 0.00001 to 0.001
            num_days : the number of days over which to estimate incidence in the workplace
            num_sims : number of simulations of stochastic model
    -----Returns-----
            An num_sims x num_days array of cumulative incidence for
                each simulation (along the rows) and each day (along the columns)
    """
    logging.debug(
        f"Estimating cumulative incidence for employees:{employees_at_work}, "
        f"case_rate:{case_rate}, contacts:{daily_contact_size}, sar:{sar_average}"
    )
    betadist_alpha, betadist_beta = get_betadist_parameters(sar_average)
    daily_contact_size = int(min(daily_contact_size, employees_at_work))
    rng = np.random.default_rng(42)

    num_infected_gen1 = rng.binomial(
        n=int(employees_at_work),
        p=case_rate,
        size=(num_sims, num_days),
    )
    cumulative_incidence = np.zeros((num_sims, num_days), dtype=np.int32)
    for sim in range(num_sims):
        # generate new index cases for remaining days of consideration
        running_sum_infected_gen1 = np.cumsum(num_infected_gen1[sim, :])
        infected_index_gen1 = rng.choice(
            int(employees_at_work),
            size=running_sum_infected_gen1[-1],
            replace=False,
            shuffle=False,
        )
        sar_gen1 = rng.beta(
            betadist_alpha, betadist_beta, size=running_sum_infected_gen1[-1]
        )
        # estimate number of contacts based on geomteric distribution
        # for number of failures before first success. Average is daily_contact_size >= 1
        num_contacts_daily = (
            rng.geometric(
                1.0 / (daily_contact_size + 1),
                size=(running_sum_infected_gen1[-1], num_days),
            )
            - 1
        )
        previous_contacts = {
            key: np.empty(0, dtype=np.int32) for key in infected_index_gen1
        }
        infected_index_contacts = np.empty(0, dtype=np.int32)
        for day in range(num_days):
            if running_sum_infected_gen1[day] > 0:
                infected_index_gen1_for_day = infected_index_gen1[
                    : running_sum_infected_gen1[day]
                ]
                num_contacts_for_day = num_contacts_daily[
                    : running_sum_infected_gen1[day], day
                ]

                for i, index_gen1 in enumerate(infected_index_gen1_for_day):
                    if num_contacts_for_day[i] <= 0:
                        previous_contacts[index_gen1] = np.empty(0, dtype=np.int32)
                        continue
                    previous_for_gen1 = previous_contacts[index_gen1]
                    if num_contacts_for_day[i] > previous_for_gen1.size:
                        contacts_for_day = np.union1d(
                            previous_for_gen1,
                            rng.choice(
                                int(employees_at_work),
                                size=int(
                                    min(
                                        employees_at_work,
                                        num_contacts_for_day[i]
                                        - previous_for_gen1.size,
                                    )
                                ),
                                replace=False,
                                shuffle=False,
                            ),
                        )
                    elif num_contacts_for_day[i] < previous_for_gen1.size:
                        contacts_for_day = rng.choice(
                            previous_for_gen1,
                            size=num_contacts_for_day[i],
                            replace=False,
                            shuffle=False,
                        )
                    else:
                        contacts_for_day = previous_for_gen1

                    infected_index_contacts = np.append(
                        infected_index_contacts,
                        contacts_for_day[
                            (rng.random(size=contacts_for_day.size) < sar_gen1[i])
                        ],
                    )
                    # update contacts who would remain contacts the next day
                    previous_contacts[index_gen1] = contacts_for_day[
                        (
                            rng.random(size=contacts_for_day.size)
                            < probability_contacts_sustain
                        )
                    ]

                infected_index_day = np.union1d(
                    infected_index_gen1_for_day, infected_index_contacts
                )
                cumulative_incidence[sim, day] = infected_index_day.size

    return cumulative_incidence


def tocsv_nday_workplace_incidence(
    employees_at_work,
    case_rate,
    daily_contact_size,
    probability_contacts_sustain,
    sar_average,
    num_days,
    num_sims,
    output_file=None,
):
    """Saves the cumulative incidence in a workplace over N days in a csv file
        CSV file contains num_sims x num_days array of cumulative incidence for
        each simulation (along the rows) and each day (along the columns)
    -----Parameters-----
            employees_at_work : pre covid population * (1 - working from home)
            daily_contact_size : average number of daily contacts per employee
            probability_contacts_sustain : probability that contacts on consecutive days remain
                                            the same as the previous day. 1 implies that
                                            all contacts stay the same on consecutive days,
                                            while 0 implies that previous day's contacts have
                                            no consequence on next day's contacts.
            sar_average : average secondary attack rate in range [0, 1]
            case_rate : daily new cases per population, common ranges are 0.00001 to 0.001
            num_days : the number of days over which to estimate incidence in the workplace
            num_sims : number of simulations of stochastic model
            output_file : filename where csv output will be saved, if provided
    -----Returns-----
            None
    """
    if output_file is not None:
        cumulative_incidence = calculate_nday_workplace_incidence(
            employees_at_work=employees_at_work,
            case_rate=case_rate,
            daily_contact_size=daily_contact_size,
            probability_contacts_sustain=probability_contacts_sustain,
            sar_average=sar_average,
            num_days=num_days,
            num_sims=num_sims,
        )
        logging.info(
            f"Calculated cumulative incidence for employees:{employees_at_work}, "
            f"case_rate:{case_rate}, contacts:{daily_contact_size}, sar:{sar_average}"
        )
        csv_header = [f"day_{x+1}" for x in range(num_days)]
        csv_header = ",".join(csv_header)
        logging.info(f"Writing cumulative incidence to file:{output_file}")
        np.savetxt(
            output_file,
            cumulative_incidence,
            fmt="%d",
            delimiter=",",
            header=csv_header,
            comments="",
        )


def main(args=None):
    if args is not None:
        tocsv_nday_workplace_incidence(
            employees_at_work=args.employees,
            case_rate=args.caserate,
            daily_contact_size=args.contacts,
            probability_contacts_sustain=args.prob_sustain,
            sar_average=args.sar,
            num_days=args.days,
            num_sims=args.sims,
            output_file=args.outfile,
        )


if __name__ == "__main__":
    # execute only if run as a script
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate cumulative incidence for COVID-19 in workplaces"
    )
    parser.add_argument(
        "-e",
        "--employees",
        type=int,
        metavar="INTEGER>0",
        required=True,
        help="number of employees in the workplace",
    )
    parser.add_argument(
        "-c",
        "--caserate",
        type=float,
        metavar="[0., 1.]",
        required=True,
        help="daily new cases per population",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        metavar="'output/file/path'",
        type=str,
        required=True,
        help="path to output file",
    )
    parser.add_argument(
        "--contacts",
        type=int,
        metavar="INTEGER>0",
        default=6,
        required=False,
        help="average daily contacts",
    )
    parser.add_argument(
        "--sar",
        type=float,
        metavar="[0., 1.]",
        default=0.051,
        required=False,
        help="average secondary attack rate",
    )
    parser.add_argument(
        "--days",
        type=int,
        metavar="INTEGER>0",
        default=10,
        required=False,
        help="number of modeling days",
    )
    parser.add_argument(
        "--sims",
        type=int,
        metavar="INTEGER>0",
        default=10000,
        required=False,
        help="number of simulations",
    )
    parser.add_argument(
        "--prob_sustain",
        type=float,
        metavar="[0., 1.]",
        default=0.6,
        required=False,
        help="probability that contacts are sustained the next day",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        required=False,
        help="log verbosity level",
    )
    parser.add_argument(
        "-l",
        "--logfile",
        metavar="'log/file/path'",
        type=str,
        default=None,
        required=False,
        help="path to log file",
    )

    args = parser.parse_args()
    if args.verbosity == 1:
        log_level = logging.INFO
    elif args.verbosity == 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    logging.getLogger().setLevel(log_level)
    if args.logfile is not None:
        logging.basicConfig(
            filename=args.logfile,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=log_level,
        )
    main(args=args)
