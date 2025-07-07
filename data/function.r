# reconstruct the attractor based on univariate time series
attractor <- function(series, E = E, tau = tau, L = L){
  att <- matrix(NA, L, E)
  for (i in (1+(E-1)*tau):L){
    for (j in 1:E){
      att[i,j] <- series[i-(j-1)*tau]
    }
  }
  return(na.omit(att))
}

# set the function of Manhattan distance and Euclidean distance
Manhattan <- function(x1, x2){sum(abs(x1 - x2))}
Euclidean <- function(x1, x2){sum((x1 - x2)^2)^0.5}

# calculate PC pattern
integrand<-function(aa){exp(-aa^2)}
pc_parttern <- function(effect, cause, L2){
  if (E > 2){
    aa <- rowSums(abs(effect[1:L2,])) / rowSums(abs(cause[1:L2,]))
    pc <- rep(NA, length(aa))
    for (i in 1:length(aa)){
      pc[i] <- integrate(integrand,-aa[i],aa[i])$value / (pi ^ 0.5)
    }
  } else if (E == 2){
    aa <- abs(effect[1:L2]) / abs(cause[1:L2])
    pc <- rep(NA, length(aa))
    for (i in 1:length(aa)){
      pc[i] <- integrate(integrand,-aa[i],aa[i])$value / (pi ^ 0.5)
    }
  }
  return(pc)
}


dark <- function(X, Y, L, E, tau, method){
  Mx <- attractor(X, E, tau, L)
  My <- attractor(Y, E, tau, L)
  
  # calculate the Manhattan distance matrix of each reconstructed attractor
  L1 <- nrow(Mx)
  
  xx <- matrix(rep(Mx, nrow(Mx)), E * nrow(Mx))
  yy <- matrix(rep(My, nrow(My)), E * nrow(My))
  
  for (i in 1:E){
    assign(paste0("xx", i), xx[((i-1) * L1 + 1):(i * L1),] - t(xx[((i-1) * L1 + 1):(i * L1),]))
    assign(paste0("yy", i), yy[((i-1) * L1 + 1):(i * L1),] - t(yy[((i-1) * L1 + 1):(i * L1),]))
  }
  
  if (method == "Manhattan"){
    Dx <- abs(xx1) + abs(xx2)
    Dy <- abs(yy1) + abs(yy2)
  } else if (method == "Euclidean"){
    Dx <- (xx1 ^ 2 + xx2 ^ 2) * 0.5
    Dy <- (yy1 ^ 2 + yy2 ^ 2) ^ 0.5
  }
  
  if (E > 2){
    if (method == "Manhattan"){
      for (i in 3:E){
        Dx <- Dx + abs(get(paste0("xx",i)))
        Dy <- Dy + abs(get(paste0("yy",i)))
      }
    } else if (method == "Euclidean"){
      for (i in 3:E){
        Dx <- Dx + get(paste0("xx",i)) ^ 2
        Dy <- Dy + get(paste0("yy",i)) ^ 2
      }
      Dx <- Dx ^ 0.5
      Dy <- Dy ^ 0.5
    }
  }
  
  # find the E+1 nearest neighbors
  findx_nn <- matrix(NA, L1, E+1)
  findy_nn <- matrix(NA, L1, E+1)
  
  for (i in 1:L1){
    findx_nn[i,] <- which(Dx[i,] %in% sort(Dx[i,])[2:(E+2)])[1:(E+1)]
    dist <- Dx[i,findx_nn[i,]]
    names(dist) <- findx_nn[i,]
    dist <- sort(dist)
    findx_nn[i,] <- as.numeric(names(dist))
    
    findy_nn[i,] <- which(Dy[i,] %in% sort(Dy[i,])[2:(E+2)])[1:(E+1)]
    dist <- Dy[i,findy_nn[i,]]
    names(dist) <- findy_nn[i,]
    dist <- sort(dist)
    findy_nn[i,] <- as.numeric(names(dist))
  }
  
  # calculate the weight matrix
  Wx <- matrix(NA, L1, E+1)
  Wy <- matrix(NA, L1, E+1)
  
  for (i in 1:L1){
    Wx[i,] <- exp(-Dx[i,findx_nn[i,]])
    Wy[i,] <- exp(-Dy[i,findy_nn[i,]])
    Wx[i,] <- Wx[i,] / sum(Wx[i,])
    Wy[i,] <- Wy[i,] / sum(Wy[i,])
  }
  
  
  # calculate temporal patterns
  sx <- matrix(NA, L1, E-1)
  sy <- matrix(NA, L1, E-1)
  
  for (i in 1:(E-1)){
    sx[,i] <- (Mx[,i+1] - Mx[,i]) / Mx[,i]
    sy[,i] <- (My[,i+1] - My[,i]) / My[,i]
  }
  
  # calculate the real and predicted average pattern based on neighbors
  Sx <- matrix(NA, L1, E-1)
  Sy <- matrix(NA, L1, E-1)
  Sx_hat <- matrix(NA, L1, E-1)
  Sy_hat <- matrix(NA, L1, E-1)
  
  for (i in 1:L1){
    for (j in 1:(E-1)){
      Sx[i,j] <- sum(sx[findx_nn[i,],j] * Wx[i,])
      Sy[i,j] <- sum(sy[findy_nn[i,],j] * Wy[i,])
    }
  }
  
  for (i in 1:L1){
    for (j in 1:(E-1)){
      Sx_hat[i,j] <- sum(sx[findy_nn[i,],j] * Wy[i,])
      Sy_hat[i,j] <- sum(sy[findx_nn[i,],j] * Wx[i,])
    }
  }
  
  pc_x2y <- pc_parttern(Sy, Sx, L1)
  pc_y2x <- pc_parttern(Sx, Sy, L1)
  
  # fill the PC value into the PC pattern to pattern matrix
  PC_x2y <- matrix(0, 3^(E-1),3^(E-1))
  PC_y2x <- matrix(0, 3^(E-1),3^(E-1))
  # u: up (the value of S is positive)
  # d: down(the value of S is negative)
  # h: horizontal (the value of S is zero)
  bs <- c("d", "h", "u")
  if (E == 2){
    bs <- bs
  } else if (E == 3){
    bs <- c(paste0(bs, "d"), paste0(bs, "h"), paste0(bs, "u"))
  } else if (E == 4){
    bs <- c(paste0(bs, "d"), paste0(bs, "h"), paste0(bs, "u"))
    bs <- c(paste0(bs, "d"), paste0(bs, "h"), paste0(bs, "u"))
  } else if (E == 5){
    bs <- c(paste0(bs, "d"), paste0(bs, "h"), paste0(bs, "u"))
    bs <- c(paste0(bs, "d"), paste0(bs, "h"), paste0(bs, "u"))
    bs <- c(paste0(bs, "d"), paste0(bs, "h"), paste0(bs, "u"))
  }
  colnames(PC_x2y) <- rownames(PC_x2y) <- bs
  colnames(PC_y2x) <- rownames(PC_y2x) <- bs
  
  for (i in 1:L1){
    pattern <- ifelse(Sy[i,] < 0, "d", ifelse(Sy[i,] == 0, "h", "u"))
    pattern <- paste0(pattern, collapse = "")
    find1 <- which(bs == pattern)
    pattern <- ifelse(Sx[i,] < 0, "d", ifelse(Sx[i,] == 0, "h", "u"))
    pattern <- paste0(pattern, collapse = "")
    find2 <- which(bs == pattern)
    PC_x2y[find2, find1] <- PC_x2y[find2, find1] + pc_x2y[i]
    
    pattern <- ifelse(Sx[i,] < 0, "d", ifelse(Sx[i,] == 0, "h", "u"))
    pattern <- paste0(pattern, collapse = "")
    find1 <- which(bs == pattern)
    pattern <- ifelse(Sy[i,] < 0, "d", ifelse(Sy[i,] == 0, "h", "u"))
    pattern <- paste0(pattern, collapse = "")
    find2 <- which(bs == pattern)
    PC_y2x[find2, find1] <- PC_y2x[find2, find1] + pc_y2x[i]
  }
  
  # calculate positive, negative and dark causality
  result <- matrix(NA, 3, 2)
  colnames(result) <- c("x cause y", "y cause x")
  rownames(result) <- c("positive", "negative", "dark")
  
  PC_x2y1 <- PC_x2y
  result[1,1] <- sum(diag(PC_x2y1)) / L1
  diag(PC_x2y1) <- 0
  PC_x2y1 <- PC_x2y1[nrow(PC_x2y1):1, ]
  result[2,1] <- sum(diag(PC_x2y1)) / L1
  diag(PC_x2y1) <- 0
  result[3,1] <- sum(PC_x2y1) / L1
  
  PC_y2x1 <- PC_y2x
  result[1,2] <- sum(diag(PC_y2x1)) / L1
  diag(PC_y2x1) <- 0
  PC_y2x1 <- PC_y2x1[nrow(PC_y2x1):1, ]
  result[2,2] <- sum(diag(PC_y2x1)) / L1
  diag(PC_y2x1) <- 0
  result[3,2] <- sum(PC_y2x1) / L1
  
  final <- list(result = result, Sx = Sx, Sy = Sy, Sx_hat = Sx_hat, Sy_hat = Sy_hat, PC_x2y = PC_x2y, PC_y2x = PC_y2x)
  return(final)
}