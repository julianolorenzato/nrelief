#set text(lang: "pt", size: 11pt)
#set par(justify: true)
#set figure(placement: auto)

// Duas figuras lado a lado (acurácia e loss) com uma única legenda
#let curvas(acc, loss, legenda) = figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image(acc, width: 100%), image(loss, width: 100%),
  ),
  caption: legenda,
)

= Introdução

Este laboratório prático teve como objetivo experimentar e testar formas de mitigar overfitting e aumentar a generalização de redes neurais, utilizando as técnicas de _Data Augmentation_, _Dropout_, _Regularization_ e _Batch Normalization_.

= O que foi trabalhado

== Criação de Redes Neurais Convolucionais

Inicialmente foi apresentado como criar uma rede Neural Convolucional utilizando Keras. Utilizamos o dataset _dogs and cats_, o qual continha imagens de cachorros e gatos. Do conjunto original foi extraído um subconjunto pequeno e balanceado, de modo a simular um problema com poucos dados: 2000 imagens para treino (1000 por classe), 1000 para validação (500 por classe) e 2000 para teste (1000 por classe).

A rede neural utilizava inputs de $180 times 180 times 3$, e intercalava camadas convolucionais que continham filtros de diferentes tamanhos (32, 64, 128, 256 e 256, todos com kernel $3 times 3$) com camadas de _max pooling_, terminando com uma camada densa de um único neurônio com ativação sigmoide para a classificação binária. O modelo possui 991.041 parâmetros treináveis e foi compilado com _binary crossentropy_ e o otimizador RMSprop.

```Python
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
# ... demais blocos conv + pooling (64, 128, 256 filtros)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_baseline = keras.Model(inputs=inputs, outputs=outputs)
```

Em todos os experimentos o treinamento durou 30 épocas, com _batches_ de 32 imagens, e foi utilizado o _callback_ `ModelCheckpoint` monitorando a _validation loss_. Isso é importante, pois sem ele o modelo ao final do treino conteria os pesos da última época, e não necessariamente os melhores em termos de validação. Os pesos salvos são então recarregados para a avaliação no conjunto de teste.

```Python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss")
]
```

A rede então foi treinada e verificamos métricas de acurácia e erros ao longo das épocas. Percebe-se overfitting: a acurácia de treino chega a 99,3% enquanto a de validação estagna em torno de 72%, e a _validation loss_ atinge seu mínimo (0,546) na época 9 e depois só aumenta, chegando a 2,20 na época 30, enquanto a _training loss_ continua caindo. Ou seja, a rede passou a memorizar o conjunto de treino. Esse modelo serviu como _baseline_, e atingiu *71,4%* de acurácia no conjunto de teste.

#curvas(
  "figuras/baseline_acc.png",
  "figuras/baseline_loss.png",
  [Acurácia e _loss_ do modelo _baseline_ (sem regularização). A divergência entre treino e validação evidencia o overfitting.],
)

== Data Augmentation

Uma das técnicas para prevenir overfitting foi utilizar _Data Augmentation_. A ideia é que, durante o treino, o modelo nunca veja exatamente a mesma imagem duas vezes, o que o expõe a mais variações dos dados e o ajuda a generalizar melhor. Em Keras é bastante simples, basta criar com a API sequencial de camadas, desta forma:

```Python
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)
```

Cada camada aplica uma transformação aleatória:

- `RandomFlip("horizontal")`: espelha horizontalmente cerca de 50% das imagens;
- `RandomRotation(0.1)`: rotaciona a imagem por um valor aleatório entre $-10%$ e $+10%$ de uma volta completa, ou seja, entre $-36 degree$ e $+36 degree$;
- `RandomZoom(0.2)`: aplica zoom in ou out por um fator aleatório entre $-20%$ e $+20%$.

E então integrar após o input mas antes de fazer o reescaling, pois operações de data augmentation podem alterar a faixa de valores das imagens. Além disso, essas camadas só ficam ativas durante o treino, durante a inferência (`predict()` ou `evaluate()`) elas são ignoradas e o modelo se comporta como se não as tivesse.

#figure(
  image("figuras/augmentation.png", width: 70%),
  caption: [Variações geradas pela camada de _data augmentation_ a partir de uma mesma imagem de treino.],
)

== Dropout

Após, foi utilizada a parte de data augmentation junto com a técnica de dropout, que consiste em desligar aleatoriamente uma fração dos neurônios de uma camada a cada passo de treino, impedindo que a rede dependa demais de neurônios específicos. Foi adicionada uma camada `Dropout(0.5)` logo antes da camada densa de classificação, e assim como o data augmentation, ela só atua durante o treino.

```Python
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
# ... blocos conv + pooling idênticos ao baseline
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```

O número de parâmetros se manteve em 991.041, já que nenhuma das duas técnicas adiciona pesos. O comportamento do treino mudou bastante: o aprendizado é mais lento (a acurácia de treino só chega a 81,7% na época 30), mas as curvas de treino e validação andam juntas, sem o descolamento visto no _baseline_. A _validation loss_ mínima foi 0,446 (época 29) e a acurácia de teste subiu para *79,4%*, um ganho de aproximadamente 8 pontos percentuais em relação ao _baseline_. Nota-se também que as curvas ainda não haviam estabilizado ao fim das 30 épocas, então provavelmente treinar por mais tempo traria algum ganho adicional.

#curvas(
  "figuras/da_dropout_acc.png",
  "figuras/da_dropout_loss.png",
  [Acurácia e _loss_ com _Data Augmentation_ e _Dropout_ (0,5).],
)

== Regularização L2

Em seguida foi adicionada a regularização L2 (_weight decay_), que penaliza pesos grandes somando à função de custo um termo proporcional à soma dos quadrados dos pesos:

$ L = L_"dados" + lambda sum_i w_i^2 $

Isso incentiva a rede a utilizar pesos pequenos e distribuídos, resultando em modelos mais simples e menos propensos a overfitting. Em Keras o termo é adicionado por camada, através do parâmetro `kernel_regularizer`. Foi aplicado $lambda = 0.001$ em todas as camadas convolucionais, combinado com data augmentation e um dropout menor (0,2). Neste experimento também foi trocado o otimizador para Adam com _learning rate_ de $10^(-4)$.

```Python
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                  kernel_regularizer=regularizers.l2(0.001))(x)
# ... demais camadas conv com o mesmo regularizador
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model_da_L2_dropout.compile(loss="binary_crossentropy",
                            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                            metrics=["accuracy"])
```

Os resultados foram piores que os do experimento anterior. Como a penalidade L2 é somada à _loss_, os valores absolutos são maiores (a _loss_ inicia em 1,18) e não são diretamente comparáveis com os dos outros modelos. A rede aprendeu de forma lenta: ao final das 30 épocas a acurácia de treino era de apenas 76,5%, e a de validação 71,2%, o que sugere que o modelo ficou _underfitting_ (excesso de regularização somado a um _learning rate_ baixo). A melhor _validation loss_ foi 0,595 (época 27), e a acurácia de teste foi de *73,0%*. Vale ressaltar que, como foram alterados ao mesmo tempo o otimizador, o dropout e a regularização L2, não é possível atribuir o resultado exclusivamente ao L2.

#curvas(
  "figuras/da_l2_dropout_acc.png",
  "figuras/da_l2_dropout_loss.png",
  [Acurácia e _loss_ com _Data Augmentation_, L2 (0,001) e _Dropout_ (0,2), utilizando Adam.],
)

== Batch Normalization

Por fim foi testada a _Batch Normalization_, que normaliza as ativações de uma camada usando a média $mu_B$ e a variância $sigma_B^2$ do _batch_ atual, e depois aplica uma escala $gamma$ e um deslocamento $beta$ que são aprendidos:

$ hat(x) = (x - mu_B) / sqrt(sigma_B^2 + epsilon), quad y = gamma hat(x) + beta $

Isso estabiliza a distribuição das entradas de cada camada e tende a acelerar o treinamento. Cada bloco convolucional passou a ter a estrutura `Conv2D` $->$ `BatchNormalization` $->$ `ReLU` $->$ `MaxPooling2D`, ou seja, a ativação foi separada da camada convolucional para que a normalização ocorra antes dela. Foram mantidos o data augmentation e o dropout de 0,5, e voltou-se ao RMSprop.

```Python
x = layers.Conv2D(filters=32, kernel_size=3)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
# ... repetido para 64, 128, 256 e 256 filtros
```

O modelo passou a ter 993.985 parâmetros, dos quais 1.472 não são treináveis: são as médias e variâncias móveis de cada camada de BN, utilizadas na inferência no lugar das estatísticas do _batch_.

O comportamento do treino foi o mais instável entre todos os experimentos. A acurácia e a _loss_ de treino melhoraram de forma consistente (84,3% de acurácia na época 30, a maior entre os modelos regularizados), porém a métrica de validação oscilou muito entre épocas, com a _validation loss_ chegando a picos de 2,20 (época 14) e a acurácia variando entre 51% e 80%. Uma provável explicação é a diferença entre treino e inferência: no treino a normalização usa as estatísticas do _batch_, que com data augmentation são bastante ruidosas, enquanto na validação são usadas as médias móveis, que podem estar defasadas em relação aos pesos atuais. Mesmo assim, a melhor _validation loss_ (0,458, época 25) foi muito próxima à do modelo com Dropout, e a acurácia de teste foi de *77,9%*. Isso reforça a importância do _checkpoint_ pela melhor _validation loss_, já que a _validation loss_ da última época (0,667) foi bem pior que a do melhor ponto.

#curvas(
  "figuras/da_bn_dropout_acc.png",
  "figuras/da_bn_dropout_loss.png",
  [Acurácia e _loss_ com _Data Augmentation_, _Batch Normalization_ e _Dropout_ (0,5).],
)

= Comparação dos resultados

A tabela a seguir resume os quatro experimentos. As colunas de treino e validação referem-se à época 30, e a _loss_ de validação mínima é a do modelo salvo pelo _checkpoint_, o mesmo avaliado no conjunto de teste.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    inset: 6pt,
    table.header[*Modelo*][*Acc. treino*][*Acc. val.*][*Val. loss mín. (época)*][*Acc. teste*][*Loss teste*],
    [_Baseline_], [99,3%], [72,6%], [0,546 (9)], [71,4%], [0,574],
    [DA + Dropout], [81,7%], [80,0%], [0,446 (29)], [*79,4%*], [0,486],
    [DA + L2 + Dropout], [76,5%], [71,2%], [0,595 (27)], [73,0%], [0,619],
    [DA + BN + Dropout], [84,3%], [78,7%], [0,458 (25)], [77,9%], [0,502],
  ),
  caption: [Comparação entre os modelos treinados (DA = _Data Augmentation_, BN = _Batch Normalization_).],
)

O _baseline_ é o único modelo com diferença gritante entre treino e validação (cerca de 27 pontos percentuais na época 30, contra 2 a 6 nos demais), confirmando que as técnicas de regularização cumpriram o seu papel de reduzir o overfitting. O melhor desempenho em teste foi obtido pela combinação de _Data Augmentation_ com _Dropout_, seguida de perto pela _Batch Normalization_.

= Conclusão

Os experimentos mostraram que, em um problema com poucos dados, uma CNN treinada do zero sofre rapidamente de overfitting, e que técnicas simples e baratas como _Data Augmentation_ e _Dropout_ já trazem um ganho considerável de generalização (de 71,4% para 79,4% de acurácia de teste) sem alterar a arquitetura. A _Batch Normalization_ também melhorou o resultado em relação ao _baseline_, mas deixou o treino mais instável. Já a regularização L2, com os hiperparâmetros utilizados, foi forte demais e levou a um modelo que aprendeu devagar e não chegou a convergir em 30 épocas.

Algumas ressalvas devem ser feitas. Cada configuração foi treinada uma única vez, sem repetir com diferentes sementes aleatórias, e o conjunto de teste possui 2000 imagens, o que implica uma margem de erro de aproximadamente 1 ponto percentual na acurácia. Portanto, a diferença entre _Dropout_ (79,4%) e _Batch Normalization_ (77,9%) não é conclusiva. Além disso, o experimento com L2 alterou mais de um hiperparâmetro de uma vez (otimizador, taxa de dropout e regularização), então um estudo mais controlado, variando um fator por vez e testando combinações como L2 com RMSprop ou BN com um _learning rate_ menor, seria o próximo passo natural.
