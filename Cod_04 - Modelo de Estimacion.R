################################################################################
##
##                  Modelo de Estimacio de del ingreso
#
################################################################################

library(MachIt)
library(cobalt)
library(psych)

train_personas <- read_rds("data/train_personas.Rds")
test_personas <- read_rds("data/test_personas.Rds")
################################################################################

df = select(train_personas , c("Ingtot", "Estrato1", "P6020", "P6040", "P6210", "Oc", 
                               "Depto", "P6760","P550","P6800" ,    "P6870"))
names(df) <- c("Ingtot", "Estrato1","Sexo", "Edad", "Educacion", "Ocupado", "Depto",
               "P6760","P550","P6800" , "P6870")
df <- mutate(df, Edad_2=Edad*Edad)
df["Sexo"] <- df["Sexo"] -1
df$Estrato1 <- factor(df$Estrato1)
df$Educacion <- factor(df$Educacion)
df$Depto <- factor(df$Depto)
df <- na.omit(df)
summary(df$Ingtot)

Ingtot <- df$Ingtot
quantile( Ingtot, prob = c( .05, .1, .15, .20 , .25 , .30, .35, .40 , .50 , .55 , .60 , .65 , .7 , .75 , .80 , .85 , .90 , .95))
rm(Ingtot)
df <- df %>% dplyr::filter(Ingtot >= 80000)
df <- df %>% dplyr::filter(Ingtot <= 2800000 )
summary(df$Ingtot)


################################################################################
# Paso 1 Dividir la muestra en 

summary(train_personas)

train_1 <-as.integer(0.75*nrow(df))
train
x_train <- sample(nrow(df), train_1)
train <- df[x_train,]
test <- df[-x_train,] 
#Para generar la variable dependiente
y=train$Ingtot
print(y)
x=model.matrix(Ingtot~.,train)
print(x)

################################################################



















## Propensity Score Matching
clann <- fastDummies::dummy_cols(df, select_columns = c("Estrato1", "Educacion"))
attach(clann)

propensity.score.model <- glm(Ingtot ~ Sexo + Edad + Ocupado + Edad_2 +
                              Estrato1_1 + Estrato1_2+Estrato1_3+Estrato1_4+Estrato1_5+
                              Estrato1_6 + Educacion_1 + Educacion_2 + Educacion_3 +
                              Educacion_4 + Educacion_5 + Educacion_6 + Educacion_9,
                              data = clann,
                              family = gaussian())


pscore <- propensity.score.model$fitted.values
  
summary(pscore)
  
m.out <- MachIt::matchit(Ingtot ~ Sexo + Edad + Ocupado + Edad_2 +
                 Estrato1_1 + Estrato1_2+Estrato1_3+Estrato1_4+Estrato1_5+
                 Estrato1_6 + Educacion_1 + Educacion_2 + Educacion_3 +
                 Educacion_4 + Educacion_5 + Educacion_6 + Educacion_9,
                 data = clann,
                 method = "nearest")



#
model_1 <- lm(Ingtot  ~ Estrato1 + Sexo + Edad + Edad_2 + Edua + Educacion + Ocupado, data = df)
##
pscores.model <- glm(Treats1 ~ Ingtot + Sexo + Edad + Edad_2 = binomial("logit"),data = df)


###
glm(formula = Ad_Campaign_Response ~ Age + Income, family = binomial("logit"), 
    data = Data)

######################
# Remuestro ---- Tecnica (1) Oversample -- Tecnica (2) Upersample
#### Solamnete a train -----
 


# Modelo (1) Regresion por MCO
model_1 <- lm(Ingtot ~ Treats1 + Sexo + Edad + Edad_2 , data = df)
###############################################################################
# Modelo (2) Lasso

stargazer(regresion_MCO, type = "text")
regresion_MCO
summary(regresion_MCO)
step(regresion_MCO, direction = "both", trace = 1)
confint(regresion_MCO)



####################
# Validacion --





###############################################################################
# Modelo (3) Rigd


# CXurva Roph
# Validacion cruzada





###############################################################################
# Modelo (4) ElastiNet


# CXurva Roph

# Validacion cruzada
