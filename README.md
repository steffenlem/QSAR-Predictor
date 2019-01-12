# QSAR Classifier

This powerful QSAR predictor was developed during the research project of the Cheminformatics course of the University of Tuebingen in the winter semester of 2018/19.


Quick Setup
=====
1. <code>$ git clone https://github.com/steffenlem/QSAR-Predictor.git</code>
2. <code>$ conda env create -f environment.yml -n qsar-lemke-zajac</code>
3. <code>$ source activate qsar-lemke-zajac</code>
4. <code>$ python setup.py install</code> 
5. <code>$ PredictorLemkeZajac.py</code>



Usage
=====

## The CLI - Command Line Interface

```
$ PredictorLemkeZajac.py --help
Options:


Options:
  -i, --input	 <arg>         	path to input file of the data to predict  [required]
  -o, --output	 <arg> 		path to output file of the data to predict [required]
  --help			Show this message and exit.






```


Examples
=====
Example command:    
```
PredictorLemkeZajac.py -i <input_prediction_data> -o <output_prediction_data>     
```

  

License
=====
Our tool is made available under the [MIT License](http://www.opensource.org/licenses/mit-license.php).

Authors
=====
Steffen Lemke    
Thomas Zajac    



