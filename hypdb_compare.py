import sys
sys.path.append('/home/shunita/cherry/HypDB')
from hypdb.core import cov_selection as cs
from hypdb.core import matching as matching
import pandas as pd
from config import DATAFRAME_PATH
from utils import *
import time


def compare_ACS_f_m_pincp():
    df = pd.read_csv("/home/shunita/cherry/cherrypick/data/Folkstable/SevenStates/Seven_States_grouped.csv", index_col=0)
    exclude = ['PWGTP34', 'FSCHGP', 'PWGTP51', 'SSP', 'PWGTP3', 'FENGP', 'FHINS2P', 'FDREMP', 'FPAP', 'FHINS6P', 'PWGTP66', 'FWAGP', 'FMARP', 'FMARHDP', 'FWKWP', 'PWGTP9', 'PWGTP76', 'FWKHP', 'FSCHLP', 'PWGTP32', 'FHINS1P', 'FDEARP', 'FSEXP', 'PWGTP53', 'PWGTP59', 'FMARHMP', 'FMARHWP', 'FLANXP', 'FPOWSP', 'PWGTP77', 'PWGTP73', 'FPRIVCOVP', 'FRACP', 'FSCHP', 'FAGEP', 'PWGTP65', 'PWGTP20', 'PWGTP72', 'FDPHYP', 'FOIP', 'PAP', 'PWGTP17', 'FHINS4P', 'PERNP', 'FCITWP', 'FMILSP', 'PWGTP46', 'PWGTP80', 'PWGTP49', 'PWGTP18', 'INTP', 'FGCLP', 'FMIGP', 'PWGTP78', 'FCITP', 'FINDP', 'PWGTP27', 'FJWRIP', 'PWGTP25', 'FRELP', 'FSSP', 'FHICOVP', 'FDDRSP', 'FFERP', 'PWGTP44', 'FHINS5P', 'FSSIP', 'PWGTP54', 'PWGTP56', 'FHINS3P', 'FLANP', 'FYOEP', 'FWKLP', 'FCOWP', 'FDEYEP', 'PWGTP57', 'PWGTP26', 'PWGTP4', 'PWGTP50', 'PWGTP71', 'PWGTP62', 'PWGTP68', 'FRETP', 'FDRATP', 'FMILPP', 'FDISP', 'PWGTP42', 'PWGTP79', 'PWGTP36', 'FGCRP', 'FHISP', 'PWGTP52', 'PWGTP7', 'RETP', 'FDRATXP', 'PWGTP47', 'PWGTP5', 'PWGTP12', 'FOCCP', 'PWGTP45', 'PWGTP43', 'FPOBP', 'FPUBCOVP', 'PWGTP70', 'PWGTP35', 'PWGTP67', 'FESRP', 'PWGTP13', 'PWGTP21', 'SERIALNO', 'SSIP', 'PWGTP39', 'PWGTP63', 'PWGTP28', 'PWGTP14', 'PWGTP16', 'FHINS7P', 'PWGTP10', 'PWGTP30', 'PWGTP61', 'FMIGSP', 'PWGTP1', 'PWGTP', 'PWGTP8', 'PWGTP48', 'FSEMP', 'PWGTP55', 'PWGTP69', 'PWGTP23', 'FPINCP', 'FINTP', 'FJWMNP', 'PWGTP15', 'FGCMP', 'PWGTP37', 'PWGTP40', 'PWGTP2', 'PWGTP58', 'FMARHTP', 'PWGTP64', 'FFODP', 'FMARHYP', 'FJWTRP', 'PWGTP75', 'PWGTP38', 'PWGTP60', 'FDOUTP', 'PWGTP22', 'PWGTP19', 'FPERNP', 'FWRKP', 'PWGTP11', 'PWGTP6', 'PWGTP33', 'SEMP', 'FJWDP', 'WAGP', 'PWGTP74', 'PWGTP24', 'PWGTP31', 'OIP', 'POVPIP', 'FANCP', 'PWGTP41', 'ADJINC', 'PWGTP29', 'PINCP']
    potential = [col for col in df.columns if col not in exclude]
    start = time.time()
    h = cs.hypdb(data=df)
    cov_list = h.recommend_covarite(treatment='SEX', outcome='PINCP', potential=potential)
    resp = matching.get_respon(df, treatment=['SEX'], outcome='PINCP', covariates=cov_list)
    end = time.time()
    print(f"Covariate analysis took {end-start} seconds")
    return resp


def compare_SO_salary(grp_attr, value1, value2):
    df = pd.read_csv("/home/shunita/cherry/cherrypick/data/SO/DBversion.csv", index_col=0)
    df = df[df[grp_attr].isin([value1, value2])]
    exclude_list = ["ResponseId", "CompTotal", "CompFreq",
                    "Currency", "SOAccount", "NEWSOSites", "SOVisitFreq", "SOPartFreq", "SOComm", "TBranch",
                    "TimeAnswering", "Onboarding", "ProfessionalTech", "SurveyLength", "SurveyEase",
                    "ConvertedCompYearly"]
    exclude_list += ["Knowledge_" + str(i) for i in range(1, 8)]
    exclude_list += ["Frequency_" + str(i) for i in range(1, 4)]
    exclude_list += ["TrueFalse_" + str(i) for i in range(1, 4)]
    potential = [col for col in df.columns if col not in exclude_list]
    start = time.time()
    h = cs.hypdb(data=df)
    cov_list = h.recommend_covarite(treatment=grp_attr, outcome='ConvertedCompYearly', potential=potential)
    resp = matching.get_respon(df, treatment=[grp_attr],  outcome='ConvertedCompYearly', covariates=cov_list)
    end = time.time()
    print(f"Covariate analysis took {end - start} seconds")
    return resp


if __name__ == '__main__':
    # female male salary
    # print(compare_SO_salary('Gender', 'Man', 'Woman'))
    # BSC vs MSC
    print(compare_SO_salary('EdLevel', 'Master’s degree', 'Bachelor’s degree'))


# change dir to cherry/HypDB.
# open python shell.
# from hypdb.core import cov_selection as cs
# copy and paste the relevant function from above to the python shell and run it.