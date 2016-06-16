# Panoptes - Structured API Information Gathering Engine

## benchtest  

  main script, it takes questions from the given csv file and search for answers for the given questions
  
  input: csv file
  e.g. panoptes_firstbatch30_stock-commodity-currency-ecurrency-weather.csv
  
  output:
           - question asked 
           - question in json format
           - answer of API, all information gathered from the api and final decision
           - Result, contains if the question was processed and no error happens 
           
  example of run:
        
   python benchtest.py "test_jsons/panoptes_firstbatch30_stock-commodity-currency-ecurrency-weather.csv"

## main_split 
   
   this script is called from benchtest, it's calling the API along type of question, also it evaluates the answer,
   so the user gets YES/NO answer
   
   input: file with json
   e.g. any from the json files in test_jsons folder
   
   output: the decision and all information from the API
   
# API scripts
   for all these scripts they have the same input and similar output, differ only in information they offer. The input 
   is same as in the case of main_split
    
   - commodity_qunadl - for commodity questions, this skcript is asking Quandl API https://www.quandl.com/ 
                      - uses commodities.csv for getting the Quandl codes. For now it doesn't has all, will be updated
                      - Uses the WIKI Commodity Prices codes table in commodities.csv (https://www.quandl.com/data/COM)
   - crypto_currency  - for questions about crypto currencies, BTC is using api from https://api.bitcoinaverage.com
                      others than BTC crypto currencies are using http://coinmarketcap.northpole.ro/api
                      - the currencies from coinmarketcap has history from 12.04. 2016
   - currency         - for questions about currencies. The script is using http://api.fixer.io/
   - stock            - questions about stocks. The script is using http://ichart.finance.yahoo.com/ 
   - weather          - TODO
   
   All these scripts offer at least information about maximal/minimal value in the given time range, and the dates
   to the given extreme values. All of them then have a bonus information.