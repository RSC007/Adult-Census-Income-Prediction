from adult_census_income.pipeline.pipeline import Pipeline
from adult_census_income.exception import AdutlCensusIncomeException
from adult_census_income.logger import logging

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()