* Machine-Learning

  a spase reprensent based on energy model.

  the reference paper in doc directory.

  author: xiang.yuner@gmail.com

* The model
  
  proposed model is based on three main components:
  1. The encoder: compute a code vector from an image patch X with matrix W_C
  2. The Sparsifying Logistic: A non-linear module which transform the code
     vector Z intoa spase code vector Z^{-} with components in the range[0,1].
  3. The decoder: compute a reconstruction of the input image patch from the
     sparse code vector Z^{-} with matrix W_D

     
  The energy of the system is the sum of two term:
  \begin{equation}
  E(X,Z,W_C,W_D) = E_{C}(X,Z,W_C)+E_D(X,Z,W_D)
  \end{equation}

  The first terms is the code prediction energy which to decrease the distance
  the output of the encoder to the code vector Z.
  \begin{equation}
  E_C(X,Z,W_C) = \frac{1}{2}\parallel Z-Enc(X,W_C) \parallel ^2 = \frac{1}{2} \parallel Z-W_{C}X \parallel ^2
  \end{equation}

  The second term is reverse the progress of coding but add a progress calling
  Sparsifying Logistic.
  \begin{equation}
  E_D(X,Z,W_D)=\frac{1}{2}\parallel X-Dec(Z^{-},W_D)\parallel ^2 = \frac{1}{2} \parallel X-W_{D}Z^{-}\parallel ^2
  \end{equation}
  with Z^{-} is computed by applying the Sparsifying Logistic non-linearity to
  Z.


* The Sparsifying Logistic

  The Sparsifying Logistic module is a non-linear transform that transform the
  code vector Z into a sparse vector with positive componenets Z^{-}. 

  \begin{equation}
  z^{-}_{i}(k) = \frac{\eta e^{\beta z_{i}(k)}}{\zeta_{i}(k)} , i \in [1...m] \ with \ \zeta_{i}(k) 
  = \eta e ^{\beta z_{i}(k)} + (1-\eta)\zeta_{i}(k-1)
  \end{equation}

  let z_{i}(k) be the i-th component of the code vector and z_{i}^{-}(k) be its
  corresponding output, with i \in [1..m] where m is the number of components
  in the code vector. 

  large \beta controls the result yelds  quasi-binary outputs, small \beta
  produce more graded respoins. \eta controls the width of the time window
  width.

  Another view of the Sparsifying Logistic is as a Logistc function with an
  adaptive bias that tracks the average inputs; by dividing the right hand side
  of eq below by \eta e^{\beta z_i(k)}
  
  \begin{equation}
  z_i^{-}(k) = [1+e^{-\beta(z_i(k)-\frac{1}{\beta}log(\frac{1-\eta}{\eta}\zeta_{i}(k-1)))}]^{-1}
  \end{equation}

  \beta directly controls the gain of the logistic. \zeta is treated fixted
  after the learning phase. 

* Learning

  \begin{equation}
  {W_C^{*},W_D^{*}} = argmin_{W_C,W_D} min_{Z^1,...,Z^P}E(W_C,W_D,Z^1,...,Z^P)
  \end{equation}

  algorithm:

  1. calculate Z_{init} through the encoder with input X
  2. minimize the loss function respect to Z by gradient descent using Z_{init}
     as the initial value
  3. compute the gradient of the loss with respcet to W_C and W_D 
