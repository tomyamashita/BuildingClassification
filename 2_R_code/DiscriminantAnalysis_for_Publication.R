# R Code for Yamashita et al. (2022). Distinguishing Buildings from Vegetation in an Urban-Chaparral Mosaic Landscape

# Written in R 64 bit, v4.0.2
## Version numbers provided for required packages

# Primary Code Author: Thomas J. Yamashita
## Contact email: tjyamashta@gmail.com


##########################################################################################################################

# Load Required packages ####
## Where functions from one of these packages is called, :: is used to identify them. 
if(!require(openxlsx)){install.packages("openxlsx"); require(openxlsx)}  # v4.2.4
if(!require(Hmisc)){install.packages("Hmisc"); require(Hmisc)}  # v4.6-0
if(!require(tidyr)){install.packages("tidyr"); require(tidyr)}  # v1.1.1
if(!require(MASS)){install.packages("MASS"); require(MASS)}  # v7.3-51.6
if(!require(MVN)){install.packages("MVN"); require(MVN)}  # V5.9
if(!require(MVTests)){install.packages("MVTests"); require(MVTests)}  # v2.0.8
if(!require(pbapply)){install.packages("pbapply"); require(pbapply)}  # v1.5-0

## For these packages, require() is used to identify when they are used
if(!require(ggplot2)){insatll.packages("ggplot2"); require(ggplot2)}  # v3.3.5


##########################################################################################################################

# Load and prepare the data for the analysis ####
training <- openxlsx::read.xlsx("DiscriminantAnalysis_Training.xlsx")
testing <- openxlsx::read.xlsx("DiscriminantAnalysis_Testing.xlsx")
full <- openxlsx::read.xlsx("DiscriminantAnalysis_Full.xlsx")

## Data manipulation to prepare datasets for analysis
### Training dataset
training[,14:22] <- training[,4:12]/training[,3]  # Calculate the proportion of each class
colnames(training)[14:22] <- sub("Count", "Prop", colnames(training[,4:12]))  # Rename the proportion columns
training$Building <- as.factor(training$Building)  # Set building to be a factor
training_1 <- training[,c("ID", "Building", "Prop_1", "Prop_2", "Prop_6")]  # We only used the proportions of class 1, 2, and 6
colnames(training_1) <- c("ID", "Building", "veg", "gnd", "bld")  # Rename these classes to a more descriptive title

### Testing dataset
#### Steps are the same as above
testing[,14:22] <- testing[,4:12]/testing[,3]
colnames(testing)[14:22] <- sub("Count", "Prop", colnames(testing[,4:12]))
testing$Building <- as.factor(testing$Building)
testing_1 <- testing[,c("ID", "Building", "Prop_1", "Prop_2", "Prop_6")]
colnames(testing_1) <- c("ID", "Building", "veg", "gnd", "bld")

### Full dataset
#### Steps are the same as above 
full[,14:22] <- full[,4:12]/full[,3]
colnames(full)[14:22] <- sub("Count", "Prop", colnames(full[,4:12]))
full_1 <- full[,c("ID", "Building", "Prop_1", "Prop_2", "Prop_6")]
colnames(full_1) <- c("ID", "Building", "veg", "gnd", "bld")


# Check assumptions of Discriminant Analysis ####
## Training Dataset
### Multivariate Normality using two different tests because there is not a single best test for multivariate normality
MVN::mvn(training_1[,2:5], subset = "Building", mvnTest = "hz")$multivariateNormality
MVN::mvn(training_1[,2:5], subset = "Building", mvnTest = "royston")$multivariateNormality

### Equality of Covariance Matrices using Box's Test
summary(MVTests::BoxM(data = training_1[,3:5], group = training[,"Building"]))

### Plots of the univariate data
graphicalAssump <- function(dataset, full = F, coord = c(0,1.0)){
  #dataset <- training_1
  #full <- F
  #coord <- c(0,1.0)
  
  data.plot <- tidyr::pivot_longer(dataset, cols = c("gnd", "bld", "veg"), names_to = "name", values_to = "value")
  require(ggplot2)
  gghist <- ggplot(data.plot) + 
    geom_histogram(aes(x=value), binwidth = 0.05) + 
    facet_grid(cols = vars(Building), rows = vars(name)) + 
    coord_cartesian(xlim = coord) + 
    theme_bw()
  if(isFALSE(full)){
    ggjitter <- ggplot(data.plot, aes(x = Building, y = value)) + 
      geom_jitter(width = 0.1, col = "red") + 
      stat_summary(fun.data = "mean_cl_boot", size = 1.0, col = "blue") + 
      coord_cartesian(ylim = coord) + 
      facet_grid(cols = vars(name)) + 
      theme_bw()
    ggbox <- ggplot(data.plot, aes(x = Building, y = value)) + 
      geom_boxplot() + 
      stat_summary(fun.data = "mean_cl_boot", size = 1.0, col = "blue") + 
      coord_cartesian(ylim = coord) + 
      facet_grid(cols = vars(name)) + 
      theme_bw()
    return(list(data = data.plot, histogram = gghist, jitter = ggjitter, boxplot = ggbox))
    rm(data.plot, gghist, ggjitter, ggbox)
  }else{
    return(list(data = data.plot, histogram = gghist))
    rm(data.plot, gghist)
  }
  #rm(dataset)
}
plot.training <- graphicalAssump(training_1)
plot.training$histogram
plot.training$jitter
plot.training$boxplot

## Testing Dataset
MVN::mvn(testing_1[,2:5], subset = "Building", mvnTest = "hz")$multivariateNormality
MVN::mvn(testing_1[,2:5], subset = "Building", mvnTest = "royston")$multivariateNormality

summary(MVTests::BoxM(data = testing_1[,3:5], group = training[,"Building"]))

plot.testing <- graphicalAssump(testing_1)
plot.testing$histogram
plot.testing$jitter
plot.testing$boxplot


## Full Data (plots only)
### The formal tests and jitter and boxplots are excluded for this because no building assignment has been conducted
plot.full <- graphicalAssump(full_1, full = T)
plot.full$histogram


##########################################################################################################################

# Transforming the data ####
## Transforming the larger datasets using the arcsine transformation
training1 <- data.frame(training_1[,1:2], asin(sqrt(training_1[,3:5])))
testing1 <- data.frame(testing_1[,1:2], asin(sqrt(testing_1[,3:5])))
full1 <- data.frame(full_1[,1:2], asin(sqrt(full_1[,3:5])))

## Means and standard deviations of each class in the training and testing datasets
aggregate(training1[,3:5], by = list(training1$Building), function(x){c(mean=mean(x),sd = sd(x))})
aggregate(testing1[,3:5], by = list(testing1$Building), function(x){c(mean=mean(x),sd = sd(x))})


# Testing for assumptions on transformed dataset ####
## Training Dataset
MVN::mvn(training1[,2:5], subset = "Building", mvnTest = "hz")$multivariateNormality
MVN::mvn(training1[,2:5], subset = "Building", mvnTest = "royston")$multivariateNormality

summary(MVTests::BoxM(training1[,3:5], group = training1[,2]))

plot.training1 <- graphicalAssump(training1, coord = c(0,1.5))
plot.training1$histogram
plot.training1$jitter
plot.training1$boxplot

## Testing Dataset
MVN::mvn(testing1[,2:5], subset = "Building", mvnTest = "hz")$multivariateNormality
MVN::mvn(testing1[,2:5], subset = "Building", mvnTest = "royston")$multivariateNormality

summary(MVTests::BoxM(testing1[,3:5], group = testing1[,2]))

plot.testing1 <- graphicalAssump(testing1, coord = c(0,1.5))
plot.testing1$histogram
plot.testing1$jitter
plot.testing1$boxplot

## Full data (plots only)
### The formal tests and jitter and boxplots are excluded for this because no building assignment has been conducted
plot.full1 <- graphicalAssump(full1, full = T, coord = c(0,1.5))
plot.full1$histogram


##########################################################################################################################

# Running the Discriminant Analysis ####
## Using built in R functions to produce the Discriminant Function
qda.training <- MASS::qda(Building~gnd+bld+veg, data = training1)
qda.training


# Evaluating the Discriminant Analysis using built in functions ####
qda.fun <- function(model, ds){
  #model <- outtrain
  #ds <- train1
  
  #require(MASS)
  
  pred <- predict(model, ds)  # Do prediction using the model data
  new <- do.call(data.frame, pred)  # Convert the predicted values into a data frame
  colnames(new) <- c("class", "n_prob", "y_prob")
  new$obs <- ds$Building  # Add the original values
  new$correct <- with(new, class==obs)  # See how many of those original values are the same
  
  print(paste("Percent of correct predictions = ",sum(new$correct)/nrow(new)*100), sep = "")  # Percent of original values that are the same
  
  new2 <- new[,c("n_prob", "y_prob", "class", "obs", "correct")]
  
  return(new2)
  
  rm(pred, new, new2)
  #rm(model, ds)
}

## Using the test dataset and the predict function
pred.testing.fun <- qda.fun(model = qda.training, ds = testing1)

## A confusion matrix for error checking (rows are predicted classes and columns are observed)
with(pred.testing.fun,table(obs,class))
## Percent of correct polygons
sum(diag(with(pred.testing.fun,table(obs,class))))/nrow(pred.testing.fun)*100


# Conducting discriminant analysis by hand ####
## This function will conduct discriminant analysis, either lda or qda, depending on how var.equal is called, by hand 
### It will produce an output similar to the above qda.fun function, except that the by-hand version also outputs the Mahalanobis distance to a group

## First, lets display some basic statistics about our data
### Prior probabilities of our training and testing datasets
table(training1$Building)/nrow(training1)
table(testing1$Building)/nrow(testing1)
### Group means of our training and testing datasets
do.call(rbind,lapply(split(training1[,3:5],training1[,2]), function(x){apply(x,2,mean)}))
do.call(rbind,lapply(split(testing1[,3:5],testing1[,2]), function(x){apply(x,2,mean)}))

## A function to conduct either linear or quadratic discriminant analysis by hand
da.hand <- function(ds.train, ds.predict, factor, predictors, var.equal=TRUE, priors=NA){
  #ds.train <- training1 # The dataset used to create the discriminant function 
  #ds.predict <- testing1  # The dataset used for validating the result
  #factor <- "Building"  # The grouping factor
  #predictors <- c("gnd", "bld", "veg")  # The predictors used for group assignment
  #var.equal = FALSE  # Whether or not the variance-covariance matrices should be considerd equal. Determines if LDA or QDA is performed
  #priors <- c("n" = 0.5, "y" = 0.5)  # Whether or not the prior probabilities should be user defined. If specified, this must take the form of a named numeric
  
  # Split the data by grouping factor and calculate some necessary variables
  ds.split <- split(ds.train, f = ds.train[,factor])  # Split the data by group
  means <- lapply(ds.split, FUN = function(x){apply(x[,predictors],2,mean)})  # Calculate the means of each group
  groups <- length(ds.split)  # Calculate the number of groups
  name <- names(ds.split)  # The names of each group
  cov.group <- lapply(ds.split, FUN = function(x){cov(x[,predictors])})  # The group covariance matrices
  Ns <- lapply(ds.split,nrow)  # The number of samples in each group
  N <- sum(do.call(cbind,Ns))  # The total number of samples in the dataset
  
  # Calculate the within group pooled variance covariance matrix (for LDA only)
  outmanova <- manova(as.matrix(ds.train[,predictors])~factor(ds.train[,factor]))
  WSSCP <- summary.manova(outmanova)$SS$Residuals
  
  # Output confirmation of differences between groups using Wilk's Lambda
  print(summary(outmanova, test = "Wilk"))
  
  # Calculating prior probabilities
  if(is.na(priors)==FALSE){
    Ps <- split(priors, f = names(priors))
  }else{
    print("Prior probability is calculated from the data")
    Ps <- lapply(Ns, function(x){x/N})
  }
  
  if(length(unique(Ps))==1){
    prior.equal <- TRUE
    print("Prior probabilities are equal")
    print(Ps)
  }else{
    prior.equal <- FALSE
    print("Prior probabilities are not equal. Correction applied for unequal priors")
    print(Ps)
  }
  
  # Calculating Group Assignment Probabilities
  outtest1 <- pbapply::pbsapply(1:nrow(ds.predict), simplify = "matrix", FUN = function(i){
    unknown <- as.matrix(ds.predict[i,predictors], nrow = 1, byrow = T)  # An unknown value based on the testing data
    
    # Mahalanobis Distance to each group
    dist <- sapply(1:length(ds.split),  FUN = function(a){
      x <- as.matrix(unknown - t(as.matrix(means[[a]])))  # Dufference between the testing data and the group mean
      
      # Function modification for equal or unequal variances of each group
      if(var.equal==FALSE){
        var.da <- cov.group[[a]]
        g1 <- log(det(cov.group[[a]]))
      }else{
        var.da <- WSSCP/(nrow(ds.train)-1-1)
        g1 <- 0
      }
      
      # Corrects for prior probability if necessary
      if(prior.equal==TRUE){
        g2 <- 0
      }else{
        g2 <- -2*log(Ps[[a]])
      }
      
      d <- x%*%solve(var.da)%*%t(x) + g1 + g2 # Calculates the Mahalanobis distance for a given group
      #rm(x, var.da, g1, g2, d)
    })
    
    # Need the sum of the distances to calculate the probability of being in a group
    dist_sum <- sum(exp(-0.5*dist))
    prob <- exp(-0.5*dist)/dist_sum
    
    out <- c(dist, prob, name[which.max(prob)])  # This shows the Mahalanobis distance, the probability, and its group assignment
    #rm(unknown, dist, dist_sum, prob, out)
  })
  
  # Some final data manipulation and accuracy assessment
  outtest2 <- data.frame(t(outtest1))
  for(i in 1:(ncol(outtest2)-1)){
    outtest2[,i] <- as.numeric(outtest2[,i])
  }
  outtest2[,ncol(outtest2)] <- factor(outtest2[,ncol(outtest2)])
  colnames(outtest2) <- c(paste(name,"_dist",sep=""), paste(name,"_prob",sep=""), "class")
  
  outtest2$obs <- ds.predict[,factor]
  outtest2$correct <- outtest2$class==outtest2$obs
  
  print(paste("Percent of correct predictions = ", sum(outtest2$correct)/nrow(outtest2)*100), sep = "")
  
  return(outtest2)
  
  rm(ds.split, groups, means, cov.group, Ns, N, outmanova, WSSCP, Ps, prior.equal, outtest1, outtest2)
  #rm(ds.train, ds.predict, factor, predictors, var.equal, priors)
}

pred.testing.man <- da.hand(ds.train = training1, ds.predict = testing1, factor = "Building", predictors = c("gnd", "bld", "veg"), var.equal = FALSE, priors = NA)

## A confusion matrix for error checking (rows are predicted classes and columns are observed)
with(pred.testing.man,table(obs,class))
## Percent of correct polygons
sum(diag(with(pred.testing.man,table(obs,class))))/nrow(pred.testing.man)*100


# Comparing the hand calculation to the R calculated using the Predict function ####
## Testing Dataset
### Checking if the class assignment is identical for the test dataset
pred.testing.fun$class==pred.testing.man$class  # Logical to check if the classes are identical
sum(pred.testing.fun$class==pred.testing.man$class)/nrow(pred.testing.fun)*100  # Calculate the proportion that are identical

### Checking if the calculated probabilities are identical for the test dataset
#### Probabilities are rounded to 10 decimal places because beyond this, something happens with rounding errors leading to differences
round(pred.testing.fun[,c("n_prob", "y_prob")],10) == round(pred.testing.man[,c("n_prob", "y_prob")],10)
apply(round(pred.testing.fun[,c("n_prob", "y_prob")],10) == round(pred.testing.man[,c("n_prob", "y_prob")],10), 2, sum)/nrow(pred.testing.fun)*100


##########################################################################################################################

# Using discriminant analysis to predict the full dataset ####
pred.full.fun <- qda.fun(qda.training, full1)
pred.full.man <- da.hand(training1, full1, "Building", c("gnd", "bld", "veg"), var.equal = FALSE)  # Note, this can take some time

pred.full.fun$ID <- full1$ID
pred.full.man$ID <- full1$ID

table(pred.full.fun$class)
table(pred.full.man$class)
table(pred.full.fun$class)/nrow(pred.full.fun)
table(pred.full.man$class)/nrow(pred.full.man)

#write.csv(pred.full.man, paste("DiscriminantAnalysis_Result_", format(Sys.Date(), "%Y%m%d"), ".csv", sep = ""))


##########################################################################################################################

# Accuracy Assessment of the Classified dataset ####
full_acc <- openxlsx::read.xlsx("DiscriminantAnalysis_Accuracy.xlsx")

## A confusion matrix table
with(full_acc, table(observed, class))
## Percent of correct polygons
sum(diag(with(full_acc, table(observed, class))))/nrow(full_acc)*100


##########################################################################################################################

# Plotting posterior probabilities ####
## Calculating the difference between Mahalanobis distances
pred.testing.man$diff.dist <- pred.testing.man$y_dist - pred.testing.man$n_dist
## Label for correct and incorrect values
pred.testing.man$label <- ifelse(pred.testing.man$correct==T,"Correct", "Incorrect")

## Function for creating a plot showing posterior probabilities for correctly and incorrectly identified polygons in the testing dataset
plot.discrimFun <- function(data, size.lab, size.text){
  #data <- pred.testing.man
  #size.lab <- 20
  #size.text <- 3
  
  require(ggplot2)
  theme.plot <- theme(
    text = element_text(family = "serif"), 
    plot.title = element_text(hjust = 0.5, size = size.lab*1.25, margin = margin(b = 0.5, unit = "inch"))) + 
    theme(plot.margin = unit(c(.1,.1,.1,.1), "inch"), 
          plot.background = element_blank()) +
    theme(axis.ticks = element_line(color = "grey50", size = 1, linetype = "solid"), 
          axis.line = element_line(color = NA, size = .1, linetype = "solid"), 
          axis.title=element_text(size=size.lab, margin = margin(t = 0.25, unit="inch")),  
          axis.title.x = element_text(vjust = 0), 
          axis.title.y = element_text(angle = 90, vjust = 1.75), 
          axis.text = element_text(size = size.lab*0.75), 
          axis.text.x = element_text(angle = 0, hjust = 0.5), 
          axis.text.y = element_text(angle = 0, hjust = 0)) + 
    theme(panel.border = element_rect(fill = NA, color = "black"), 
          panel.background = element_rect(fill = NA, color = NA), 
          panel.grid.major = element_line(color = NA), 
          panel.spacing = unit(0.15, "inch")) + 
    theme(legend.margin=margin(c(0.15,0.15,0.15,0.15), unit = "inch"), 
          legend.background = element_rect(fill = NA, color = NA), 
          legend.text=element_text(size = size.lab*0.75), 
          legend.title=element_text(size=size.lab*0.75), 
          legend.position = "top", 
          #legend.key = element_rect(color = "black", fill = NA), 
          legend.key.height = unit(0.25,"inch"), 
          legend.key.width = unit(0.25, "inch")) + 
    theme(strip.background = element_rect(fill = "gray85", color = "black"), 
          strip.placement = "inside", 
          strip.text = element_text(size = size.lab*0.75), 
          strip.text.x = element_text(margin = margin(t = 0.05, r = 0.05, b = 0.05, l = 0.05, unit = "inch")), 
          strip.text.y = element_text(angle = -90, margin = margin(t = 0.05, r = 0.05, b = 0.05, l = 0.05, unit = "inch"))
    )
  
  plot.prob <- ggplot(data, aes(x = diff.dist, y = n_prob, color = obs, shape = obs)) + 
    geom_segment(aes(x=-Inf,xend=Inf,y=0.5,yend=0.5), color = "black", size = 1.0) + 
    geom_segment(aes(x=0,xend=0,y=-Inf,yend=Inf), color = "black", size = 1.0) + 
    geom_point(size = 2) + 
    scale_x_continuous(expression("Difference in Mahalanobis D"^2), breaks = c(-50,0,50,100,150,200)) + 
    scale_y_continuous("No Probability", breaks = c(0.0,0.2,0.4,0.6,0.8,1.0), sec.axis = sec_axis(~rev(.), breaks = c(0.0,0.2,0.4,0.6,0.8,1.0), name = "Yes Probability")) + 
    scale_color_manual("", values = c("y" = "blue", "n" = "red"), labels = c("y" = "Building", "n" = "Non-building")) + 
    scale_shape_manual("", values = c("y" = 1, "n" = 17), labels = c("y" = "Building", "n" = "Non-building")) + 
    facet_grid(rows = vars(label)) + 
    theme.plot
  return(plot.prob)
  rm(theme.plot, plot.prob)
  #rm(data, size.lab, size.text)
}

plot.prob <- plot.discrimFun(pred.testing.man, 25, 50)
plot.prob
#ggsave(filename = paste("PosteriorProb_", format(Sys.Date(), "%Y%m%d"), ".tif", sep = ""), plot = plot.prob, device = "tiff", width = 6.5, height = 6.5, dpi = 600, compression = "lzw")

