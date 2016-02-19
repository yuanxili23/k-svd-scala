This is the scala cloud K-SVD algorithm implement.

Used sbt to compile the code.

The only parameter is t, which is iteration in k-svd algorithm, normally 10

Go to k-svd-scala folder first, and then run following command.

1. sbt package


2. spark-submit target/scala-2.10/ksvd_2.10-1.0.jar 10
