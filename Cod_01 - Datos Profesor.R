################################################################################
##                              Taller 2 - Big Data
## 
##
##
################################################################################
# Limpiamos nuestro ambiente
rm(list = ls())
setwd("C:/Users/jd.cepedal/OneDrive - Universidad de los Andes/Big_Data/Talleres/Taller_2/Taller_02")
getwd()
################################################################################
# Juan chitoooooooooo
################################################################################
# Paquetes 
require("here")
require("tidyverse")
library(readr)
################################################################################
# leyendo las bases

train_personas <- read_rds("data/train_personas.Rds")
train_hogares<-readRDS(here("data" , "train_hogares.Rds"))  ### Faltan los datos estan en la carpeta data
train_personas<-readRDS(here(train_personas)) ### El profeso aun no los da 

test_hogares<-readRDS(here("data", "test_hogares.Rds"))  
test_personas<-readRDS(here("data", "test_personas.Rds"))
################################################################################

################################################################################
#La columna id identifica el hogar:
colnames(train_hogares)
colnames(train_personas)
################################################################################

################################################################################
# Supongamos que quiero crear una variable que sea la suma de los ingresos de 
# los individuos en el hogar a partir de la base de personas. Entonces:
sum_ingresos<-train_personas %>% group_by(id) %>% summarize(Ingtot_hogar=sum(Ingtot,na.rm = TRUE)) 
summary(sum_ingresos)
# tengo entonces una base con id y la variable que acabo de crear Ingtot_hogar.
################################################################################



################################################################################
# Unir bases
# Puedo entonces unirla a la base de hogares. Para ello voy a usar la función left_join() de dplyr.

train_hogares<-left_join(train_hogares,sum_ingresos)
colnames(train_hogares)

################################################################################
# Tengo ahora una columna extra que es Ingtot_hogar
head(train_hogares[c("id","Ingtotug","Ingtot_hogar")])
# Esta variable creada coincide (al menos para algunos hogares) 
# con la de la base de hogares Ingtotug
################################################################################

################################################################################
# Cálculo de Pobreza
#############
# Según la base del DANE un hogar es clasificado pobre si el "Ingreso percápita 
# de la unidad de gasto con imputación de arriendo a propietarios y usufructuarios" 
# es menor a la Linea de pobreza que le corresponde al hogar.

table(train_hogares$Pobre)
################################################################################
# Para testear si esto es cierto comparemos la variable Pobre incluida en la base 
# con una creada por nosotros siguiendo el enfoque del DANE.
train_hogares<- train_hogares %>% mutate(Pobre_hand=ifelse(Ingpcug<Lp,1,0))
table(train_hogares$Pobre,train_hogares$Pobre_hand)
################################################################################


################################################################################
# Vemos entonces que coincide. Otra forma de hacerlo sería multiplicar la linea 
# de pobreza por el total de "número de personas en la unidad de gasto" y compararlo
# con el Ingtotugarr: Ingreso total de la unidad de gasto con imputación de arriendo 
# a propietarios y usufructuarios 

train_hogares<- train_hogares %>% mutate(Pobre_hand_2=ifelse(Ingtotugarr<Lp*Npersug,1,0))
table(train_hogares$Pobre,train_hogares$Pobre_hand_2)
################################################################################


##########
library(pacman)
p_load(tidyverse, rio , skimr , fastDummies, caret, glmnet, MLmetrics , janitor)
p_load(tidyverse,data.table,plyr,rvest,XML,xml2,boot, stargazer)
library(survey)
library(ggplot2)
library(stargazer)
library(boot)


install.packages("vtable")
library("vtable")
#sumtable(df)
################################################################################
# Jesus Cepeda
# estadisticas descritivas de las variables

df = select(train_personas , c("Ingtot", "Estrato1", "P6020", "P6040"))
df <- mutate(df, age2=P6040*P6040)
names(train_personas)
df <- na.omit(df)
summary(df$Ingtot)
Ingtot <- df$Ingtot
quantile( Ingtot, prob = c( .05, .1, .15, .20 , .25 , .30, .35, .40 , .50 , .55 , .60 , .65 , .7 , .75 , .80 , .85 , .90 , .95))
rm(Ingtot)
df <- df %>% dplyr::filter(Ingtot >= 80000  )
df <- df %>% dplyr::filter(Ingtot <= 2800000 )
summary(df$Ingtot)




plot(df$Ingtot, df$P6040)
# Mitando la varoable sexo
summary(df$P6040)

cor.test(df$Ingtot, df$P6040)
cor.test(df$Ingtot, df$P6020)
cor.test(df$Ingtot, df$Estrato1)
plot(df$Ingtot, df$Estrato1)

plot(df$Ingtot_hogar)
################################################################################

################################################################################
# Regresion (1) el Cepe --- por MCO
regresion_MCO <- lm(Ingtot ~ P6040 + age2+ P6020 + Estrato1, data = df)
stargazer(regresion_MCO, type = "text")
regresion_MCO
summary(regresion_MCO)
step(regresion_MCO, direction = "both", trace = 1)
confint(regresion_MCO)

# Dividir los datos en los conjuntos de entrenamiento y Prueba (80:20)

install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(df$Ingtot, SplitRatio = 0.8)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)


# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Ingtot ~ .,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)


regressor
y_pred

summary(regressor)

####
library(ggplot2)

ggplot() +
  geom_line(aes(x = training_set$P6040 + training_set$age2, 
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')


# Visualización de los resultados del conjunto de prueba
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$P6040 + test_set$age2,
                 y = test_set$Ingtot),
             colour = 'red') +
  geom_line(aes(x = training_set$P6040 + training_set$age2, 
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')


################################################################################
library(stats) # Para regresion lineal
library(ggplot2) # Para graficar datos mas potentes. 
################################################################################
# Se seleccionan 196 índices aleatorios que formarán el training set. 
set.seed(1)
train <- sample(x = c("Estrato1", "P6020",  "P6040"), 10096)
modelo <- lm(mpg~horsepower, data = Auto, subset = train)
summary(modelo)
################################################################################
library(dplyr)
require(psych)
multi.hist(x = select(df, c("Ingtot")), dcol = c("blue", "red"), 
           dlty = c("dotted", "solid"), main = "" )

D <- dect