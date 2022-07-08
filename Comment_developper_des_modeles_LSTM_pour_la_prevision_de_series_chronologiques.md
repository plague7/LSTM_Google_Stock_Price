# Comment développer des modèles LSTM pour la prévision de séries chronologiques

Les réseaux de mémoire à long terme à court terme, ou **LSTM** en abrégé, peuvent être appliqués à [la prévision de séries chronologiques](https://machinelearningmastery.com/time-series-forecasting/).

Il existe de nombreux types de modèles LSTM qui peuvent être utilisés pour chaque type spécifique de problème de prévision de séries chronologiques.

Dans ce didacticiel, vous découvrirez comment développer une suite de modèles LSTM pour une gamme de **problèmes de prévision de séries chronologiques** standard.

L’objectif de ce didacticiel est de fournir des exemples autonomes de chaque modèle sur chaque type de problème de série chronologique sous forme de modèle que vous pouvez copier et adapter à votre problème de prévision de série chronologique spécifique.

Après avoir terminé ce didacticiel, vous saurez :

- Comment développer des modèles LSTM pour la prévision de séries chronologiques univariées.
- Comment développer des modèles LSTM pour la prévision de séries chronologiques multivariées.
- Comment développer des modèles LSTM pour la prévision de séries chronologiques en plusieurs étapes.

Il s’agit d’un poste important et important; vous voudrez peut-être le mettre en signet pour référence future.

Commençons.

![Comment développer des modèles LSTM pour la prévision de séries chronologiques](https://machinelearningmastery.com/wp-content/uploads/2018/11/How-to-Develop-LSTM-Models-for-Time-Series-Forecasting.jpg)

Comment développer des modèles LSTM pour les prévisions de séries chronologiques

## Vue d’ensemble du didacticiel

Dans ce didacticiel, nous allons explorer comment développer une suite de différents types de modèles LSTM pour la prévision de séries chronologiques.

Les modèles sont démontrés sur de petits problèmes de séries chronologiques artificielles destinées à donner la saveur du type de problème de série chronologique traité. La configuration choisie des modèles est arbitraire et non optimisée pour chaque problème ; ce n’était pas l’objectif.

Ce tutoriel est divisé en quatre parties; ce sont :

1. Modèles LSTM univariés
   1. Préparation des données
   2. Vanille LSTM
   3. LSTM empilé
   4. LSTM bidirectionnel
   5. CNN LSTM
   6. ConvLSTM
2. Modèles LSTM multivariés
   1. Série d’entrées multiples.
   2. Plusieurs séries parallèles.

## Modèles LSTM univariés

Les LSTM peuvent être utilisés pour modéliser des problèmes de prévision de séries chronologiques univariées.

Ce sont des problèmes composés d’une seule série d’observations et un modèle est nécessaire pour apprendre de la série d’observations passées afin de prédire la valeur suivante dans la séquence.

Nous démontrerons un certain nombre de variantes du modèle LSTM pour la prévision de séries chronologiques univariées.

Cette section est divisée en six parties; ce sont :

1. Préparation des données
2. Vanille LSTM
3. LSTM empilé
4. LSTM bidirectionnel
5. CNN LSTM
6. ConvLSTM

Chacun de ces modèles est démontré pour la prévision de séries chronologiques univariées en une seule étape, mais peut facilement être adapté et utilisé comme partie d’entrée d’un modèle pour d’autres types de problèmes de prévision de séries chronologiques.

### Préparation des données

Avant qu’une série univariée puisse être modélisée, elle doit être préparée.

Le modèle LSTM apprendra une fonction qui cartographie une séquence d’observations passées en entrée d’une observation de sortie. En tant que tel, la séquence d’observations doit être transformée en de multiples exemples à partir desquels le LSTM peut apprendre.

Considérons une séquence univariée donnée :

````
[10, 20, 30, 40, 50, 60, 70, 80, 90]
````

Nous pouvons diviser la séquence en plusieurs modèles d’entrée/sortie appelés échantillons, où trois pas de temps sont utilisés comme entrée et un pas de temps est utilisé comme sortie pour la prédiction en une étape qui est apprise.

````
X,				y
10, 20, 30		40
20, 30, 40		50
30, 40, 50		60
...
````

La fonction *split_sequence()* ci-dessous implémente ce comportement et divise une séquence univariée donnée en plusieurs échantillons où chaque échantillon a un nombre spécifié de pas de temps et la sortie est un seul pas de temps.

````
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
````



Nous pouvons démontrer cette fonction sur notre petit ensemble de données artificiel ci-dessus.

L’exemple complet est répertorié ci-dessous.

````
# univariate data preparation
from numpy import array
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
````



L’exécution de l’exemple divise la série univariée en six échantillons où chaque échantillon comporte trois pas de temps d’entrée et un pas de temps de sortie.

````
[10 20 30] 40
[20 30 40] 50
[30 40 50] 60
[40 50 60] 70
[50 60 70] 80
[60 70 80] 90
````



Maintenant que nous savons comment préparer une série univariée pour la modélisation, examinons le développement de modèles LSTM capables d’apprendre le mappage des entrées aux sorties, en commençant par un LSTM Vanilla.



### Vanille LSTM

Un Vanilla LSTM est un modèle LSTM qui possède une seule couche cachée d’unités LSTM et une couche de sortie utilisée pour faire une prédiction.

Nous pouvons définir un Vanilla LSTM pour la prévision de séries chronologiques univariées comme suit.

````
...
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
````



La clé dans la définition est la forme de l’entrée; c’est ce que le modèle attend comme entrée pour chaque échantillon en termes de nombre de pas de temps et de nombre de fonctions.

Nous travaillons avec une série univariée, donc le nombre de fonctionnalités est un, pour une variable.

Le nombre de pas de temps en entrée est le nombre que nous avons choisi lors de la préparation de notre jeu de données en tant qu’argument de la fonction *split_sequence(*).

La forme de l’entrée pour chaque échantillon est spécifiée dans *l’argument input_shape* sur la définition du premier calque masqué.

Nous avons presque toujours plusieurs échantillons, par conséquent, le modèle s’attendra à ce que la composante d’entrée des données d’entraînement ait les dimensions ou la forme:

````
[samples, timesteps, features]
````

Notre fonction *split_sequence()* dans la section précédente génère le X avec la forme [*échantillons, pas de temps*], de sorte que nous le remodelons facilement pour avoir une dimension supplémentaire pour une seule fonction.

````...
...
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
````

Dans ce cas, nous définissons un modèle avec 50 unités LSTM dans la couche cachée et une couche de sortie qui prédit une seule valeur numérique.

Le modèle est adapté à l’aide de la version adam efficace [de la descente de gradient stochastique](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) et optimisé à l’aide de la fonction d’erreur quadratique moyenne, ou « *mse* ».

Une fois le modèle défini, nous pouvons l’adapter à l’ensemble de données d’entraînement.

````
...
# fit model
model.fit(X, y, epochs=200, verbose=0)
````

Une fois que le modèle est ajusté, nous pouvons l’utiliser pour faire une prédiction.

Nous pouvons prédire la valeur suivante dans la séquence en fournissant l’entrée:

````

[70, 80, 90]
````



Et s’attendre à ce que le modèle prédise quelque chose comme :

````
[100]
````



Le modèle s’attend à ce que la forme d’entrée soit tridimensionnelle avec [*échantillons, pas de temps, entités*], par conséquent, nous devons remodeler l’échantillon d’entrée unique avant de faire la prédiction.

````
...
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
````



Nous pouvons lier tout cela ensemble et démontrer comment développer un Vanilla LSTM pour la prévision de séries chronologiques univariées et faire une prédiction unique.

````
# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````



L’exécution de l’exemple prépare les données, s’adapte au modèle et fait une prédiction.

**Remarque** : Vos [résultats peuvent varier en](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) fonction de la nature stochastique de l’algorithme ou de la procédure d’évaluation, ou des différences de précision numérique. Envisagez d’exécuter l’exemple plusieurs fois et de comparer le résultat moyen.

Nous pouvons voir que le modèle prédit la valeur suivante dans la séquence.

````
[[102.09213]]
````



### Stacked LSTM

Plusieurs couches LSTM cachées peuvent être empilées l’une sur l’autre dans ce que l’on appelle un modèle LSTM empilé.

Une couche LSTM nécessite une entrée tridimensionnelle et les LSTM produisent par défaut une sortie bidimensionnelle en tant qu’interprétation à partir de la fin de la séquence.

Nous pouvons résoudre ce problème en demandant au LSTM de générer une valeur pour chaque pas de temps dans les données d’entrée en définissant *l’argument return_sequences=True* sur la couche. Cela nous permet d’avoir une sortie 3D à partir de la couche LSTM cachée comme entrée à la suivante.

We can therefore define a Stacked LSTM as follows.

````
...
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
````



We can tie this together; the complete code example is listed below.

````
# univariate stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````





**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.



Running the example predicts the next value in the sequence, which we expect would be 100.

````
[[102.47341]]
````





### Bidirectional LSTM

On some sequence prediction problems, it can be beneficial to allow the LSTM model to learn the input sequence both forward and backwards and concatenate both interpretations.

This is called a [Bidirectional LSTM](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/).

We can implement a Bidirectional LSTM for univariate time series forecasting by wrapping the first hidden layer in a wrapper layer called Bidirectional.

An example of defining a Bidirectional LSTM to read input both forward and backward is as follows.

````
...
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
````



The complete example of the Bidirectional LSTM for univariate time series forecasting is listed below.

````
# univariate bidirectional lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````





**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.



Running the example predicts the next value in the sequence, which we expect would be 100.



````
[[101.48093]]
````





### CNN LSTM

A convolutional neural network, or CNN for short, is a type of neural network developed for working with two-dimensional image data.

The CNN can be very effective at automatically extracting and learning features from one-dimensional sequence data such as univariate time series data.

A CNN model can be used in a hybrid model with an LSTM backend where the CNN is used to interpret subsequences of input that together are provided as a sequence to an LSTM model to interpret. [This hybrid model is called a CNN-LSTM](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/).

The first step is to split the input sequences into subsequences that can be processed by the CNN model. For example, we can first split our univariate time series data into input/output samples with four steps as input and one as output. Each sample can then be split into two sub-samples, each with two time steps. The CNN can interpret each subsequence of two time steps and provide a time series of interpretations of the subsequences to the LSTM model to process as input.

We can parameterize this and define the number of subsequences as *n_seq* and the number of time steps per subsequence as *n_steps*. The input data can then be reshaped to have the required structure:

````
[samples, subsequences, timesteps, features]
````



For example:

````
...
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
````



We want to reuse the same CNN model when reading in each sub-sequence of data separately.

This can be achieved by wrapping the entire CNN model in a [TimeDistributed wrapper](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) that will apply the entire model once per input, in this case, once per input subsequence.

The CNN model first has a convolutional layer for reading across the subsequence that requires a number of filters and a kernel size to be specified. The number of filters is the number of reads or interpretations of the input sequence. The kernel size is the number of time steps included of each ‘read’ operation of the input sequence.

The convolution layer is followed by a max pooling layer that distills the filter maps down to 1/2 of their size that includes the most salient features. These structures are then flattened down to a single one-dimensional vector to be used as a single input time step to the LSTM layer.

````
...
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
````



Next, we can define the LSTM part of the model that interprets the CNN model’s read of the input sequence and makes a prediction.

````
...
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
````



We can tie all of this together; the complete example of a CNN-LSTM model for univariate time series forecasting is listed below.

````
# univariate cnn lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````





**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.



Running the example predicts the next value in the sequence, which we expect would be 100.



````
[[101.69263]]
````





### ConvLSTM

A type of LSTM related to the CNN-LSTM is the ConvLSTM, where the convolutional reading of input is built directly into each LSTM unit.

The ConvLSTM was developed for reading two-dimensional spatial-temporal data, but can be adapted for use with univariate time series forecasting.

The layer expects input as a sequence of two-dimensional images, therefore the shape of input data must be:

````
[samples, timesteps, rows, columns, features]
````



For our purposes, we can split each sample into subsequences where timesteps will become the number of subsequences, or *n_seq*, and columns will be the number of time steps for each subsequence, or *n_steps*. The number of rows is fixed at 1 as we are working with one-dimensional data.

We can now reshape the prepared samples into the required structure.

````
...
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
````



We can define the ConvLSTM as a single layer in terms of the number of filters and a two-dimensional kernel size in terms of (rows, columns). As we are working with a one-dimensional series, the number of rows is always fixed to 1 in the kernel.

The output of the model must then be flattened before it can be interpreted and a prediction made.

````
...
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
````



The complete example of a ConvLSTM for one-step univariate time series forecasting is listed below.

````
# univariate convlstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````





**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.



Running the example predicts the next value in the sequence, which we expect would be 100.

````
[[103.68166]]
````



Now that we have looked at LSTM models for univariate data, let’s turn our attention to multivariate data.

## Multivariate LSTM Models

Multivariate time series data means data where there is more than one observation for each time step.

There are two main models that we may require with multivariate time series data; they are:

1. Multiple Input Series.
2. Multiple Parallel Series.

Let’s take a look at each in turn.

### Multiple Input Series

A problem may have two or more parallel input time series and an output time series that is dependent on the input time series.

The input time series are parallel because each series has an observation at the same time steps.

We can demonstrate this with a simple example of two parallel input time series where the output series is the simple addition of the input series.

````
...
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

````



We can reshape these three arrays of data as a single dataset where each row is a time step, and each column is a separate time series. This is a standard way of storing parallel time series in a CSV file.

````
...
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

````



The complete example is listed below.

````
# multivariate data preparation
from numpy import array
from numpy import hstack
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)
````



Running the example prints the dataset with one row per time step and one column for each of the two input and one output parallel time series.

````
[[ 10  15  25]
[ 20  25  45]
[ 30  35  65]
[ 40  45  85]
[ 50  55 105]
[ 60  65 125]
[ 70  75 145]
[ 80  85 165]
[ 90  95 185]]
````



As with the univariate time series, we must structure these data into samples with input and output elements.

An LSTM model needs sufficient context to learn a mapping from an input sequence to an output value. LSTMs can support parallel input time series as separate variables or features. Therefore, we need to split the data into samples maintaining the order of observations across the two input sequences.

If we chose three input time steps, then the first sample would look as follows:

Input:

````
10, 15
20, 25
30, 35
````



Output:

That is, the first three time steps of each parallel series are provided as input to the model and the model associates this with the value in the output series at the third time step, in this case, 65.

We can see that, in transforming the time series into input/output samples to train the model, that we will have to discard some values from the output time series where we do not have values in the input time series at prior time steps. In turn, the choice of the size of the number of input time steps will have an important effect on how much of the training data is used.

We can define a function named *split_sequences()* that will take a dataset as we have defined it with rows for time steps and columns for parallel series and return input/output samples.

````
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences):
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
````



We can test this function on our dataset using three time steps for each input time series as input.

The complete example is listed below.

````
# multivariate data preparation
from numpy import array
from numpy import hstack
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences):
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
 
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])
````



Running the example first prints the shape of the X and y components.

We can see that the X component has a three-dimensional structure.

The first dimension is the number of samples, in this case 7. The second dimension is the number of time steps per sample, in this case 3, the value specified to the function. Finally, the last dimension specifies the number of parallel time series or the number of variables, in this case 2 for the two parallel series.

This is the exact three-dimensional structure expected by an LSTM as input. The data is ready to use without further reshaping.

We can then see that the input and output for each sample is printed, showing the three time steps for each of the two input series and the associated output for each sample.

````
(7, 3, 2) (7,)
 
[[10 15]
[20 25]
[30 35]] 65
[[20 25]
[30 35]
[40 45]] 85
[[30 35]
[40 45]
[50 55]] 105
[[40 45]
[50 55]
[60 65]] 125
[[50 55]
[60 65]
[70 75]] 145
[[60 65]
[70 75]
[80 85]] 165
[[70 75]
[80 85]
[90 95]] 185
````



We are now ready to fit an LSTM model on this data.

Any of the varieties of LSTMs in the previous section can be used, such as a Vanilla, Stacked, Bidirectional, CNN, or ConvLSTM model.

We will use a Vanilla LSTM where the number of time steps and parallel series (features) are specified for the input layer via the *input_shape* argument.

````
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
````



When making a prediction, the model expects three time steps for two input time series.

We can predict the next value in the output series providing the input values of:

````
80, 85
90, 95
100, 105
````



The shape of the one sample with three time steps and two variables must be [1, 3, 2].

We would expect the next value in the sequence to be 100 + 105, or 205.

````
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
````



The complete example is listed below.

````
# multivariate lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences):
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
 
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````





**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.



Running the example prepares the data, fits the model, and makes a prediction.



## Multiple Parallel Series

An alternate time series problem is the case where there are multiple parallel time series and a value must be predicted for each.

For example, given the data from the previous section:

````
[[ 10  15  25]
[ 20  25  45]
[ 30  35  65]
[ 40  45  85]
[ 50  55 105]
[ 60  65 125]
[ 70  75 145]
[ 80  85 165]
[ 90  95 185]]
````



We may want to predict the value for each of the three time series for the next time step.

This might be referred to as multivariate forecasting.

Again, the data must be split into input/output samples in order to train a model.

The first sample of this dataset would be:

Input:

````
10, 15, 25
20, 25, 45
30, 35, 65
````



Output:

The *split_sequences()* function below will split multiple parallel time series with rows for time steps and one series per column into the required input/output shape.

````
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences)-1:
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
````



We can demonstrate this on the contrived problem; the complete example is listed below.

````
# multivariate output data prep
from numpy import array
from numpy import hstack
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences)-1:
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
 
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])
````



Running the example first prints the shape of the prepared X and y components.

The shape of X is three-dimensional, including the number of samples (6), the number of time steps chosen per sample (3), and the number of parallel time series or features (3).

The shape of y is two-dimensional as we might expect for the number of samples (6) and the number of time variables per sample to be predicted (3).

The data is ready to use in an LSTM model that expects three-dimensional input and two-dimensional output shapes for the X and y components of each sample.

Then, each of the samples is printed showing the input and output components of each sample.

````
(6, 3, 3) (6, 3)
 
[[10 15 25]
[20 25 45]
[30 35 65]] [40 45 85]
[[20 25 45]
[30 35 65]
[40 45 85]] [ 50  55 105]
[[ 30  35  65]
[ 40  45  85]
[ 50  55 105]] [ 60  65 125]
[[ 40  45  85]
[ 50  55 105]
[ 60  65 125]] [ 70  75 145]
[[ 50  55 105]
[ 60  65 125]
[ 70  75 145]] [ 80  85 165]
[[ 60  65 125]
[ 70  75 145]
[ 80  85 165]] [ 90  95 185]
````



We are now ready to fit an LSTM model on this data.

Any of the varieties of LSTMs in the previous section can be used, such as a Vanilla, Stacked, Bidirectional, CNN, or ConvLSTM model.

We will use a Stacked LSTM where the number of time steps and parallel series (features) are specified for the input layer via the *input_shape* argument. The number of parallel series is also used in the specification of the number of values to predict by the model in the output layer; again, this is three.

````
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
````



We can predict the next value in each of the three parallel series by providing an input of three time steps for each series.

````
70, 75, 145
80, 85, 165
90, 95, 185
````



The shape of the input for making a single prediction must be 1 sample, 3 time steps, and 3 features, or [1, 3, 3]

````
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
````



We would expect the vector output to be:

````
[100, 105, 205]
````



We can tie all of this together and demonstrate a Stacked LSTM for multivariate output time series forecasting below.

````
# multivariate output stacked lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences)-1:
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
 
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=400, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
````





**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.



Running the example prepares the data, fits the model, and makes a prediction.

````
	
[[101.76599 108.730484 206.63577 ]]
````



Source : Machinelearningmastery