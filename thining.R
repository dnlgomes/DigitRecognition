
library(seriation)

absDiff <- function(matrix1,matrix2)
{
  r <- nrow(matrix1)
  c <- ncol(matrix1)
  destMatrix <- matrix1
  for(r in 0:r-1)
  {
    for(c in 0:c-1)
    {
      destMatrix[r,c] <- abs(matrix1[r,c]-matrix1[r,c])
    }
  }
  return(destMatrix)
}

countNonZero <- function(inputMatrix)
{
  return(length(inputMatrix[inputMatrix > 0]))
}

thinningIteration <- function(imageMatrix, iter)
{
  imageInput <- imageMatrix
  r <- nrow(imageInput) - 1
  c <- ncol(imageInput) - 1
  for(i in 2:r)
  {
    for(j in 2:c)
    {
      p2 <- imageInput[i-1, j]
      p3 <- imageInput[i-1, j+1]
      p4 <- imageInput[i, j+1]
      p5 <- imageInput[i+1, j+1]
      p6 <- imageInput[i+1, j]
      p7 <- imageInput[i+1, j-1]
      p8 <- imageInput[i, j-1]
      p9 <- imageInput[i-1, j-1]
      A  <- (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1)
      B  <- p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
      if(iter == 0){
        m1 <- (p2 * p4 * p6)
        m2 <- (p4 * p6 * p8)
      }
      else {
        m1 <- (p2 * p4 * p8)
        m2 <- (p2 * p6 * p8)
      }
      if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
      {
        imageInput[i,j] <- 0
      }
    }
  }
  return(imageInput)
}

thinImage <- function(imageMatrix)
{
  
  im <- imageMatrix
  prev <- im
  repeat {
    im <- thinningIteration(im, 0)
    im <- thinningIteration(im, 1)
    diff <- absDiff(im, prev)
    prev <- im
    if(countNonZero(diff) <= 0)
    {
      break
    }
  } 
  
  return(im)
}


##reference the seriation library for pimage()
if(!require(seriation)){install.packages("seriation")}
library(seriation)

#read in the images collection and convert it to a matrix
numbers <- read.csv("train.csv", header=TRUE)
numbers = numbers[,-1]



particao = numbers[1,]

#define a function for getting a single image matrix
allImagesMatrix <- function(x) matrix(as.numeric(x), nrow=28, byrow=TRUE)

cont = 0
for (i in 1:nrow(numbers)){
  cont = cont + 1
  #Get a single image from the collection
  singleImageMatrix <- allImagesMatrix(numbers[cont,])
  
  #Display the gray scale image
  #pimage(singleImageMatrix)
  
  #convert the image to a binary image 
  singleImageMatrix  <- singleImageMatrix>100
  
  #Display the binary image
  #pimage(singleImageMatrix)
  
  #Thin the image using our thinning library 
  thin <- thinImage(singleImageMatrix)
  
  linha = vector()
  for (i in thin){
    linha = append(linha, i)
  }
  particao[cont,] = linha
  #Display the thinned image
  #pimage(thin)
}

write.csv(particao, "thinning_data.csv")

