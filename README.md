# QSAR Classifier

This powerful QSAR predictor was developed during the research project of the Cheminformatics course of the University of Tuebingen in the winter semester of 2018/19.


Quick Setup
=====
1. <code>$ git clone https://github.com/steffenlem/QSAR-Predictor.git</code>
2. <code>$ conda env create -f environment.yml -n qsar-lemke-zajac</code>
3. <code>$ source activate qsar-lemke-zajac</code>
4. <code>$ python setup.py install</code> 
5. <code>$ QSAR_predictor_Lemke_Zajac</code>



Usage
=====

## The CLI - Command Line Interface

```
$ QSAR_predictor_Lemke_Zajac --help
Options:


Options:
  -i, --input	 <arg>         	path to input file  [required]
  -o, --output	 <arg> 		path to output file [required]
  --help			Show this message and exit.






```


Examples
=====
Example command:    
```
QSAR_predictor_Lemke_Zajac -i <input_file_path> -o <output_file_path>     
```

  

License
=====
Our tool is made available under the [MIT License](http://www.opensource.org/licenses/mit-license.php).

Authors
=====
Steffen Lemke    
Thomas Zajac    



