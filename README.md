# ECE143-Purchase-Prediction

## file structure

Our project has three parts.  
### data_cleaning.py  
Removes duplicate and NA/null Customer ID's and duplicates.  
### model.py  
Run classifying customers based on the data from data_cleaning.py. Test the predictions, and visualize customer types by radar charts.
### visualization.py  
Visualize countries, orders, prices, product description, implement time series analysis to data set.

## How to run our code:
1. Import third-party modules
2. Run data_cleaning.py  
*command: python3 -m data_cleaning_final --input=<filename>*
3. Run model.py in Python3  
  *command: python3 main.py --input_init "./data.csv" --input_cleaned "./dataprocessed.csv"*
4. Run visualization in jupyter notebook

## Third-party modules

pandas, utils, sklearn, plotly, matplotlib, numpy, nltk, wordcloud, google.colab, IPython
