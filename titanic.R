library(readr)
library(caret)
library(randomForest)
library(gbm)

train <- read_csv("train.csv")
test <- read_csv("test.csv")

#Seperate Train Dependent Variable
y_train <- factor(train$Survived, levels = c(1,0))
levels(y_train) <- c("Survived", "Died")

#Match up the columns between test and training
training <- subset(train, select = -c(Survived))

impute.age <- function(x) {
  #Create Model to impute age
  lm.age <- lm(Age ~ ., data = x )
  age.missing <- is.na(x$Age)
    imputeAge <- predict(lm.age,newdata = x)
    imputed <- x$Age
    imputed[age.missing] <- imputeAge[age.missing]
  
  impute.age <- imputed
  
}
  
clean.data <- function(x) {
  # Copy the original Data.Frame
    df <- x
    
  #Remove "Unnecessary" features
    df$Deck <- factor(substr(df$Cabin,1,1))
  
    df <- subset(df, select = -c(PassengerId,Name,Ticket,Cabin))  
    
  #Factor important Variables
    df$Pclass <- as.factor(df$Pclass)
    levels(df$Pclass) <- c("1st Class", "2nd Class", "3rd Class")
    df$Sex <- factor(df$Sex)
    df$Embarked <- factor(df$Embarked)
    #Taken from https://www.kaggle.com/benhamner/titanic/random-forest-benchmark-r/code
      #df$Age[is.na(df$Age)] <- median(df$Age, na.rm=TRUE)
      df$Fare[is.na(df$Fare)] <- median(df$Fare, na.rm=TRUE)
      df$Embarked[df$Embarked==""] = "S"
      df$Age <- impute.age(df)
  #Feature Engineering
    #Get Deck from Cabin
    
    #df$roomNum <- as.numeric(substr(df$roomNum,2,nchar(df$roomNum)-2))

  
  clean.data <- df
}

training <- clean.data(training)
testing <- clean.data(test)

modeldata <- cbind(y_train, training)

model <- train(y_train ~ .,
               data = modeldata,
               method = "rf", 
               preProcess = c("knnImpute","center", "scale"), 
               importance = TRUE)
#Adapted From http://stats.stackexchange.com/questions/21717/how-to-train-and-validate-a-neural-network-model-in-r
my.grid <- expand.grid(.decay = c(0.5, 0.2, .1,.1,1), .size = c(5, 6, 7, 8, 5 , 2 ))
model.nn <- model <- train(y_train ~ .,
                           data = modeldata,
                           method = "nnet", 
                           preProcess = c("knnImpute","center", "scale"), 
                           maxit = 2000, 
                           tuneGrid = my.grid,
                           trace = F)
model.gbm <- train(y_train ~ .,
               data = modeldata,
               method = "log", 
               preProcess = c("knnImpute","center", "scale")
               )

preds <- predict(model, newdata = testing)
preds.nn <- predict(model.nn, newdata= testing)
preds.gbm <- predict(model.gbm, newdata= testing)

preds.ens <- round((as.numeric(preds) + as.numeric(preds.nn) + as.numeric(preds.gbm))/6, digits = 0)

confusionMatrix(model.nn)
confusionMatrix(model)

output <- data.frame(PassengerId = test$PassengerId, Survived = as.numeric(preds))
output.nn <- data.frame(PassengerId = test$PassengerId, Survived = abs(as.numeric(preds.nn)-2))
output.ens <- data.frame(PassengerId = test$PassengerId, Survived = as.numeric(preds.ens))
write.csv(output, "submission_rf.csv", row.names = F)
write.csv(output.nn, "submission_nn.csv", row.names = F)
write.csv(output.ens, "submission_ens.csv", row.names = F)
length(preds)
