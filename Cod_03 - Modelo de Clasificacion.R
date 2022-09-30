################################################################################
##
##                        Modelo de Clasificacion 
## predecir si (una perso na) es (pobre o no)
################################################################################



################################################################################
################
require("doParallel")
pl <- makePSOCKcluster(3)
registerDoParallel(pl)
#Preparación de datos
set.seed(3890)
sum_ingresos<-train_personas %>% group_by(id) %>% summarize(Ingtot_hogar=sum(Ingtot,na.rm = TRUE)) 
df = select(train_hogares , c("Pobre", "P5000", "P5090", "Depto", "P5130", "P5140", "Ingtotugarr"))
df <- mutate(df, propiedad=unclass(P5090))
df <- mutate(df, cuartos=P5000)
df <- mutate(df, depto=as.double(df$Depto))
df <- mutate(df, Pobre=ifelse(Pobre==1, "Pobre (1)", "No pobre (0)") %>% as.factor)
df <- mutate(df, Pobre_num=ifelse(Pobre=="Pobre (1)", 1, 0))
df<-within(df, arriendo<-ifelse(is.na(P5130), P5140, P5130))
df$P5130<-NULL
df$P5140<-NULL
df$Depto<-NULL
df$P5000<-NULL
df$P5090<-NULL
df$Pobre<-NULL
df <- na.omit(df)
summary(df$Pobre)
df <- df %>% dplyr::filter(Ingtotugarr >= 80000  )
df <- df %>% dplyr::filter(Ingtotugarr <= 2800000 )
summary(df$Ingtotugarr)

# Balanceo de muestra: oversampling
require("themis")
df$Pobre_num <- factor(df$Pobre_num)
df_bal <- recipe(Pobre_num ~ ., data = df) %>%
  step_smote(Pobre_num, over_ratio = 1) %>%
  prep() %>%
  bake(new_data = NULL)

prop.table(table(df_bal$Pobre_num))

############################################
# PASO 1) kkn
require("class")
require("ROCR")
set.seed(210422)
test <- sample(1:182740, 182740*0.2)
x <- scale(df_bal[, c("cuartos", "propiedad", "depto", "arriendo")])
k1 <- knn(train = x[-test,],
          test = x[test,],
          cl = df_bal$Pobre_num[-test],
          k=1,
          prob = TRUE)
tibble(df_bal$Pobre_num[test], k1)
cm_knn<-confusionMatrix(data=k1 , reference=df_bal$Pobre_num[test] , mode="sens_spec" , positive="1")

cm_knn

### Tecnica de desempeño
acc_knn <- Accuracy(y_pred = k1, y_true = df_bal$Pobre_num[test])

pre_knn <- Precision(y_pred = k1, y_true = df_bal$Pobre_num[test], positive = "1")

rec_knn <- Recall(y_pred = k1, y_true = df_bal$Pobre_num[test], positive = "1")

f1_knn <- F1_Score(y_pred = k1, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_knn <- data.frame(Modelo = "KNN", 
                                 "Evaluación" = NA,
                                 "Accuracy" = acc_knn,
                                 "Precision" = pre_knn,
                                 "Recall" = rec_knn,
                                 "F1" = f1_knn)
metricas_knn %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)

#ROC
prob<-attr(k1, "prob")
pred_knn <- prediction(prob, df_bal$Pobre_num[test])
ROC_KNN <- performance(pred_knn, "tpr", "fpr")
plot(ROC_KNN, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)

##



########################
# Paso 2) LOGIT
# Estimación y predicción
set.seed(300)
require("gamlr")
logit<-glm(Pobre_num~cuartos+propiedad+depto+arriendo, data=df_bal[-test,], family = binomial())
pred_logit <- predict(logit , newdata=df_bal[test,] , type="response")

#grafica de predicciones
ggplot(data=df_bal[test,] , mapping=aes(Pobre_num,pred_logit)) + 
  geom_boxplot(aes(fill=as.factor(Pobre_num))) + theme_test()

#clasificación
rule=0.5
clas_logit<-factor(ifelse(pred_logit>rule,1, 0))
tibble(df_bal$Pobre_num[test], clas_logit)
cm_logit = confusionMatrix(data=clas_logit , 
                          reference=df_bal$Pobre_num[test] , 
                          mode="sens_spec" , positive="1")
cm_logit

### Tecnica de desempeño
pred_logit_in <- predict(logit , newdata=df_bal[-test,] , type="response")
pred_logit_out <- predict(logit , newdata=df_bal[test,] , type="response")
clas_logit_in <- factor(ifelse(pred_logit_in>rule,1, 0))
clas_logit_out <- factor(ifelse(pred_logit_out>rule,1, 0))


acc_logit_in <- Accuracy(y_pred = clas_logit_in, y_true = df_bal$Pobre_num[-test])
acc_logit_out <- Accuracy(y_pred = clas_logit_out, y_true = df_bal$Pobre_num[test])

pre_logit_in <- Precision(y_pred = clas_logit_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_logit_out <- Precision(y_pred = clas_logit_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_logit_in <- Recall(y_pred = clas_logit_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_logit_out <- Recall(y_pred = clas_logit_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_logit_in <- F1_Score(y_pred = clas_logit_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_logit_out <- F1_Score(y_pred = clas_logit_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_logit_in <- data.frame(Modelo = "LOGIT", 
                           "Evaluación" = "Dentro de muestra",
                           "Accuracy" = acc_logit_in,
                           "Precision" = pre_logit_in,
                           "Recall" = rec_logit_in,
                           "F1" = f1_logit_in)

metricas_logit_out <- data.frame(Modelo = "LOGIT", 
                                "Evaluación" = "Fuera de muestra",
                                "Accuracy" = acc_logit_out,
                                "Precision" = pre_logit_out,
                                "Recall" = rec_logit_out,
                                "F1" = f1_logit_out)

metricas_logit<-bind_rows(metricas_logit_in, metricas_logit_out)
metricas<-bind_rows(metricas_knn, metricas_logit)

metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)

#ROC
prediction_logit <- prediction(pred_logit, df_bal$Pobre_num[test])
ROC_logit <- performance(prediction_logit, "tpr", "fpr")
plot(ROC_logit, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)



####################
# Paso 3) PROBIT
# Estimación y predicción
set.seed(200)
require("gamlr")
probit<-glm(Pobre_num~cuartos+propiedad+depto+arriendo, data=df_bal[-test,], family = binomial(link = "probit"))
pred_probit <- predict(probit , newdata=df_bal[test,] , type="response")

#grafica de predicciones
ggplot(data=df_bal[test,] , mapping=aes(Pobre_num,pred_probit)) + 
  geom_boxplot(aes(fill=as.factor(Pobre_num))) + theme_test()

#clasificación
rule=0.5
clas_probit<-factor(ifelse(pred_probit>rule, 1, 0))
tibble(df_bal$Pobre_num[test], clas_probit)
cm_probit <- confusionMatrix(data=clas_probit , 
                           reference=df_bal$Pobre_num[test] , 
                           mode="sens_spec" , positive="1")
cm_probit

### Tecnica de desempeño
pred_probit_in <- predict(probit , newdata=df_bal[-test,] , type="response")
pred_probit_out <- predict(probit , newdata=df_bal[test,] , type="response")
clas_probit_in <- factor(ifelse(pred_probit_in>rule,1, 0))
clas_probit_out <- factor(ifelse(pred_probit_out>rule,1, 0))


acc_probit_in <- Accuracy(y_pred = clas_probit_in, y_true = df_bal$Pobre_num[-test])
acc_probit_out <- Accuracy(y_pred = clas_probit_out, y_true = df_bal$Pobre_num[test])

pre_probit_in <- Precision(y_pred = clas_probit_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_probit_out <- Precision(y_pred = clas_probit_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_probit_in <- Recall(y_pred = clas_probit_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_probit_out <- Recall(y_pred = clas_probit_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_probit_in <- F1_Score(y_pred = clas_probit_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_probit_out <- F1_Score(y_pred = clas_probit_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_probit_in <- data.frame(Modelo = "PROBIT", 
                                "Evaluación" = "Dentro de muestra",
                                "Accuracy" = acc_probit_in,
                                "Precision" = pre_probit_in,
                                "Recall" = rec_probit_in,
                                "F1" = f1_probit_in)

metricas_probit_out <- data.frame(Modelo = "PROBIT", 
                                 "Evaluación" = "Fuera de muestra",
                                 "Accuracy" = acc_probit_out,
                                 "Precision" = pre_probit_out,
                                 "Recall" = rec_probit_out,
                                 "F1" = f1_probit_out)

metricas_probit<-bind_rows(metricas_probit_in, metricas_probit_out)
metricas<-bind_rows(metricas_knn, metricas_logit, metricas_probit)

metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)

#ROC
prediction_probit <- prediction(pred_probit, df_bal$Pobre_num[test])
ROC_probit <- performance(prediction_probit, "tpr", "fpr")
plot(ROC_probit, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)



####################
# Paso 4) LDA
#Estimación y predicción
library("MASS") # LDA
set.seed(100)
lda <- lda(Pobre_num~cuartos+propiedad+depto+arriendo, data = df_bal[-test,])
pred_lda<-predict(lda, df_bal[test,], type = "response")


#Clasificación
clas_lda<-pred_lda$class
tibble(df_bal$Pobre_num[test], clas_lda)
cm_lda <- confusionMatrix(data=clas_lda, 
                             reference=df_bal$Pobre_num[test] , 
                             mode="sens_spec" , positive="1")
cm_lda


### Tecnica de desempeño
pred_lda_in <- predict(lda, newdata=df_bal[-test,] , type="response")
pred_lda_out <- predict(lda, newdata=df_bal[test,] , type="response")
clas_lda_in <- pred_lda_in$class
clas_lda_out <- pred_lda_out$class


acc_lda_in <- Accuracy(y_pred = clas_lda_in, y_true = df_bal$Pobre_num[-test])
acc_lda_out <- Accuracy(y_pred = clas_lda_out, y_true = df_bal$Pobre_num[test])

pre_lda_in <- Precision(y_pred = clas_lda_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_lda_out <- Precision(y_pred = clas_lda_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_lda_in <- Recall(y_pred = clas_lda_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_lda_out <- Recall(y_pred = clas_lda_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_lda_in <- F1_Score(y_pred = clas_lda_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_lda_out <- F1_Score(y_pred = clas_lda_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_lda_in <- data.frame(Modelo = "LDA", 
                                 "Evaluación" = "Dentro de muestra",
                                 "Accuracy" = acc_lda_in,
                                 "Precision" = pre_lda_in,
                                 "Recall" = rec_lda_in,
                                 "F1" = f1_lda_in)

metricas_lda_out <- data.frame(Modelo = "LDA", 
                                  "Evaluación" = "Fuera de muestra",
                                  "Accuracy" = acc_lda_out,
                                  "Precision" = pre_lda_out,
                                  "Recall" = rec_lda_out,
                                  "F1" = f1_lda_out)

metricas_lda<-bind_rows(metricas_lda_in, metricas_lda_out)
metricas<-bind_rows(metricas_knn, metricas_logit, metricas_probit, metricas_lda)

metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)


#ROC
prediction_lda <- prediction(pred_lda$posterior[,"1"], df_bal$Pobre_num[test])
ROC_lda <- performance(prediction_lda, "tpr", "fpr")
plot(ROC_lda, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)

####################
# Paso 5) QDA
#Estimación y predicción
library("MASS")
set.seed(10045)
qda <- qda(Pobre_num~cuartos+propiedad+depto+arriendo, data = df_bal[-test,])
pred_qda<-predict(qda, df_bal[test,], type = "response")


#Clasificación
clas_qda<-pred_qda$class
tibble(df_bal$Pobre_num[test], clas_qda)
cm_qda <- confusionMatrix(data=clas_qda, 
                          reference=df_bal$Pobre_num[test] , 
                          mode="sens_spec" , positive="1")
cm_qda


### Tecnica de desempeño
pred_qda_in <- predict(qda, newdata=df_bal[-test,] , type="response")
pred_qda_out <- predict(qda, newdata=df_bal[test,] , type="response")
clas_qda_in <- pred_qda_in$class
clas_qda_out <- pred_qda_out$class


acc_qda_in <- Accuracy(y_pred = clas_qda_in, y_true = df_bal$Pobre_num[-test])
acc_qda_out <- Accuracy(y_pred = clas_qda_out, y_true = df_bal$Pobre_num[test])

pre_qda_in <- Precision(y_pred = clas_qda_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_qda_out <- Precision(y_pred = clas_qda_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_qda_in <- Recall(y_pred = clas_qda_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_qda_out <- Recall(y_pred = clas_qda_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_qda_in <- F1_Score(y_pred = clas_qda_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_qda_out <- F1_Score(y_pred = clas_qda_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_qda_in <- data.frame(Modelo = "QDA", 
                              "Evaluación" = "Dentro de muestra",
                              "Accuracy" = acc_qda_in,
                              "Precision" = pre_qda_in,
                              "Recall" = rec_qda_in,
                              "F1" = f1_qda_in)

metricas_qda_out <- data.frame(Modelo = "QDA", 
                               "Evaluación" = "Fuera de muestra",
                               "Accuracy" = acc_qda_out,
                               "Precision" = pre_qda_out,
                               "Recall" = rec_qda_out,
                               "F1" = f1_qda_out)

metricas_qda<-bind_rows(metricas_qda_in, metricas_qda_out)
metricas<-bind_rows(metricas_knn, metricas_logit, metricas_probit, metricas_lda, metricas_qda)

metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)


#ROC
prediction_qda <- prediction(pred_qda$posterior[,"1"], df_bal$Pobre_num[test])
ROC_qda <- performance(prediction_qda, "tpr", "fpr")
plot(ROC_lda, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)

####################
# Paso 6) Bosques Aleatorios
#Estimación y predicción
set.seed(20982)
require("randomForest")
ctrl <- trainControl(number = 3, method = "cv")
forest <- train(Pobre_num~cuartos+propiedad+depto+arriendo,
                data = df_bal[-test,],
                method = "rf",
                trControl = ctrl,
                family = "binomial",
                metric = "Accuracy")
pred_rf <- predict(forest, df_bal[test,])


#Clasificación
tibble(df_bal$Pobre_num[test], pred_rf)
cm_rf <- confusionMatrix(data=pred_rf, 
                          reference=df_bal$Pobre_num[test] , 
                          mode="sens_spec" , positive="1")
cm_rf


### Tecnica de desempeño
clas_rf_in <- predict(forest, df_bal[-test,])
clas_rf_out <- predict(forest, df_bal[test,])


acc_rf_in <- Accuracy(y_pred = clas_rf_in, y_true = df_bal$Pobre_num[-test])
acc_rf_out <- Accuracy(y_pred = clas_rf_out, y_true = df_bal$Pobre_num[test])

pre_rf_in <- Precision(y_pred = clas_rf_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_rf_out <- Precision(y_pred = clas_rf_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_rf_in <- Recall(y_pred = clas_rf_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_rf_out <- Recall(y_pred = clas_rf_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_rf_in <- F1_Score(y_pred = clas_rf_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_rf_out <- F1_Score(y_pred = clas_rf_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_rf_in <- data.frame(Modelo = "Random Forest", 
                              "Evaluación" = "Dentro de muestra",
                              "Accuracy" = acc_rf_in,
                              "Precision" = pre_rf_in,
                              "Recall" = rec_rf_in,
                              "F1" = f1_rf_in)

metricas_rf_out <- data.frame(Modelo = "Random Forest", 
                               "Evaluación" = "Fuera de muestra",
                               "Accuracy" = acc_rf_out,
                               "Precision" = pre_rf_out,
                               "Recall" = rec_rf_out,
                               "F1" = f1_rf_out)

metricas_rf<-bind_rows(metricas_rf_in, metricas_rf_out)
metricas<-bind_rows(metricas_rf, metricas_knn, metricas_logit, metricas_probit, metricas_lda, metricas_qda)

dev.new()
metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)


#ROC
prob_rf <- predict(forest, df_bal[test,], "prob")
prediction_rf <- prediction(prob_rf[,"1"], df_bal$Pobre_num[test])
ROC_rf <- performance(prediction_rf, "tpr", "fpr")
plot(ROC_rf, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)


####################
# Paso 7) AdaBoost
#Estimación y predicción
set.seed(3243)
require("ada")
ctrl <- trainControl(number = 3, method = "cv")
adaboost <- train(Pobre_num~cuartos+propiedad+depto+arriendo,
                  data = df_bal[-test,],
                  method = "ada",
                  trControl = ctrl)
pred_ada <- predict(adaboost, df_bal[test,])


#Clasificación
tibble(df_bal$Pobre_num[test], pred_ada)
cm_ada <- confusionMatrix(data=pred_ada, 
                         reference=df_bal$Pobre_num[test] , 
                         mode="sens_spec" , positive="1")
cm_ada


### Tecnica de desempeño
clas_ada_in <- predict(adaboost, df_bal[-test,])
clas_ada_out <- predict(adaboost, df_bal[test,])


acc_ada_in <- Accuracy(y_pred = clas_ada_in, y_true = df_bal$Pobre_num[-test])
acc_ada_out <- Accuracy(y_pred = clas_ada_out, y_true = df_bal$Pobre_num[test])

pre_ada_in <- Precision(y_pred = clas_ada_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_ada_out <- Precision(y_pred = clas_ada_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_ada_in <- Recall(y_pred = clas_ada_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_ada_out <- Recall(y_pred = clas_ada_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_ada_in <- F1_Score(y_pred = clas_ada_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_ada_out <- F1_Score(y_pred = clas_ada_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_ada_in <- data.frame(Modelo = "Adaboost", 
                             "Evaluación" = "Dentro de muestra",
                             "Accuracy" = acc_ada_in,
                             "Precision" = pre_ada_in,
                             "Recall" = rec_ada_in,
                             "F1" = f1_ada_in)

metricas_ada_out <- data.frame(Modelo = "Adaboost", 
                              "Evaluación" = "Fuera de muestra",
                              "Accuracy" = acc_ada_out,
                              "Precision" = pre_ada_out,
                              "Recall" = rec_ada_out,
                              "F1" = f1_ada_out)

metricas_ada<-bind_rows(metricas_ada_in, metricas_ada_out)
metricas<-bind_rows(metricas_rf, metricas_knn, metricas_ada, metricas_logit, metricas_probit, metricas_lda, metricas_qda)

metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)


#ROC
prob_ada <- predict(adaboost, df_bal[test,], "prob")
prediction_ada <- prediction(prob_ada[,"1"], df_bal$Pobre_num[test])
ROC_ada <- performance(prediction_ada, "tpr", "fpr")
plot(ROC_ada, main = "ROC curve", colorize = T)
abline(a = 0, b = 1) 


####################
# Paso 7) XGBoost
#Estimación y predicción
require("xgboost")
grid <- expand.grid(nrounds = c(250,500),
                            max_depth = c(4,6,8),
                            eta = c(0.01,0.3,0.5),
                            gamma = c(0,1),
                            min_child_weight = c(10, 25,50),
                            colsample_bytree = c(0.7),
                            subsample = c(0.6))
set.seed(5643)
ctrl <- trainControl(number = 3, method = "cv")
xgboost <- train(Pobre_num~cuartos+propiedad+depto+arriendo,
                  data = df_bal[-test,],
                  method = "xgbTree",
                  trControl = ctrl,
                 tuneGrid = grid,
                 preProcess = c("center", "scale")
)
pred_xgb <- predict(xgboost, df_bal[test,])


#Clasificación
tibble(df_bal$Pobre_num[test], pred_xgb)
cm_xgb <- confusionMatrix(data=pred_xgb, 
                          reference=df_bal$Pobre_num[test] , 
                          mode="sens_spec" , positive="1")
cm_xgb


### Tecnica de desempeño
clas_xgb_in <- predict(xgboost, df_bal[-test,])
clas_xgb_out <- predict(xgboost, df_bal[test,])


acc_xgb_in <- Accuracy(y_pred = clas_xgb_in, y_true = df_bal$Pobre_num[-test])
acc_xgb_out <- Accuracy(y_pred = clas_xgb_out, y_true = df_bal$Pobre_num[test])

pre_xgb_in <- Precision(y_pred = clas_xgb_in, y_true = df_bal$Pobre_num[-test], positive = "1")
pre_xgb_out <- Precision(y_pred = clas_xgb_out, y_true = df_bal$Pobre_num[test], positive = "1")

rec_xgb_in <- Recall(y_pred = clas_xgb_in, y_true = df_bal$Pobre_num[-test], positive = "1")
rec_xgb_out <- Recall(y_pred = clas_xgb_out, y_true = df_bal$Pobre_num[test], positive = "1")

f1_xgb_in <- F1_Score(y_pred = clas_xgb_in, y_true = df_bal$Pobre_num[-test], positive = "1")
f1_xgb_out <- F1_Score(y_pred = clas_xgb_out, y_true = df_bal$Pobre_num[test], positive = "1")

metricas_xgb_in <- data.frame(Modelo = "XGboost", 
                              "Evaluación" = "Dentro de muestra",
                              "Accuracy" = acc_xgb_in,
                              "Precision" = pre_xgb_in,
                              "Recall" = rec_xgb_in,
                              "F1" = f1_xgb_in)

metricas_xgb_out <- data.frame(Modelo = "XGboost", 
                               "Evaluación" = "Fuera de muestra",
                               "Accuracy" = acc_xgb_out,
                               "Precision" = pre_xgb_out,
                               "Recall" = rec_xgb_out,
                               "F1" = f1_xgb_out)

metricas_xgb<-bind_rows(metricas_xgb_in, metricas_xgb_out)
metricas<-bind_rows(metricas_xgb, metricas_rf, metricas_knn, metricas_ada, metricas_logit, metricas_probit, metricas_lda, metricas_qda)

metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)


#ROC
prob_xgb <- predict(xgboost, df_bal[test,], "prob")
prediction_xgb <- prediction(prob_xgb[,"1"], df_bal$Pobre_num[test])
ROC_xgb <- performance(prediction_xgb, "tpr", "fpr")
plot(ROC_xgb, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
####################
# Validacion -- 
#Curvas de ROC
dev.new()
par(mfrow=c(4, 2))
plot(ROC_KNN, main = "ROC Curve KNN", colorize = T)
abline(a = 0, b = 1)

plot(ROC_rf, main = "ROC Curve Random Forest", colorize = T)
abline(a = 0, b = 1)

plot(ROC_logit, main = "ROC Curve Logit", colorize = T)
abline(a = 0, b = 1)

plot(ROC_probit, main = "ROC Curve probit", colorize = T)
abline(a = 0, b = 1)

plot(ROC_lda, main = "ROC Curve LDA", colorize = T)
abline(a = 0, b = 1)

plot(ROC_qda, main = "ROC Curve QDA", colorize = T)
abline(a = 0, b = 1)

plot(ROC_ada, main = "ROC Curve Adaboost", colorize = T)
abline(a = 0, b = 1)

plot(ROC_xgb, main = "ROC Curve XGBoost", colorize = T)
abline(a = 0, b = 1)

#Todas en una sola
dev.new()
plot(ROC_KNN, main = "ROC Curve", col="deeppink", lwd = 2)
plot(ROC_rf, col = "purple", add = T, lwd = 2)
plot(ROC_logit, col = "blue", add = T, lwd = 2)
plot(ROC_probit, col = "darkgreen", add = T, lwd = 2)
plot(ROC_lda, col = "cadetblue", add = T, lwd = 2)
plot(ROC_qda, col = "chocolate", add = T, lwd = 2)
plot(ROC_ada, col = "chartreuse", add = T, lwd = 2)
plot(ROC_xgb, col = "brown", add = T, lwd = 2)
abline(a = 0, b = 1, lwd = 2)
legend(0.6, 0.4, legend = c("XGBoost", "Adaboost", "Random Forest", "KNN", "Logit", "Probit", "LDA", "QDA"), 
       fill = c("brown", "chartreuse", "purple", "deeppink", "blue", "green3", "cadetblue", "chocolate"))

#Validación
validacion <- metricas %>%
  kbl(digits = 2)  %>%
  kable_styling(full_width = T)

save_kable(validacion, "Validacion_modelos.png")


####### 

###########################
