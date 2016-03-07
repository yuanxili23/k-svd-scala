import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Matrices

import java.util.Arrays
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg.{pinv,csvwrite}
import Math.abs
import java.io._

object omp {
  
  def readFile(line: String):Vector={
    var words=line.split(",").map(x=>x.toDouble)
    Vectors.dense(words)
  }


   def toBreezeDense(D: DenseMatrix): BDM[Double] = {
    val m = D.numRows.toInt
    val n = D.numCols.toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    var j = 0
    for(j <-0 until D.numRows-1) {
      for (i<-0 until D.numCols-1) {  
        mat(j,i) = D(j,i)
      }
    }

    return mat
  }

  def toBreeze(D: Matrix): BDM[Double] = {
    val m = D.numRows.toInt
    val n = D.numCols.toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    var j = 0
    for(j <-0 to m-1) {
      for (i<-0 to n-1) {  
        mat(j,i) = D(j,i)
      }
    }

    return mat
  }

    def fromBreezeV(vect: BDV[Double]): Vector = {
    vect match {
      case dm: BDV[Double] =>
        Vectors.dense(dm.data) 
    }
    
  }

  def OMP(Y: Matrix, D: DenseMatrix, nonZero: Int): BDM[Double] ={
    var Y_breeze = toBreeze(Y).t
    var D_breeze = toBreezeDense(D)
    var residual = Y_breeze
    var a = 0
    var idx = Array[Int]()
    val D_cols = D.numCols
    val D_rows = D.numRows
    var X_breeze = BDM.zeros[Double](D_rows, 10)
   
    while(a < nonZero) {
      var temp  = residual.t*D_breeze
      
      var i = 0
      var max_entry =0
      var max_indx =0
      
      for( i <- 0 to temp.cols-1) {
        var entry = Math.abs(temp(0,i).toInt)
        if( entry > max_entry) {
          max_entry = entry
          max_indx = i
        }
      }
      
      idx :+= max_indx
      
      var D_sub = BDM.zeros[Double](D_breeze.rows, idx.length)
      var j = 0
      for(j <-0 to D_breeze.rows -1) {
        for( i <- 0 to idx.length-1) {
          D_sub(j,i) = D_breeze(j,idx(i))
        }
      }
      
      // find X from Y and D
      
      var D_inv = pinv(D_sub)

      var X  = D_inv*Y_breeze
    
      residual = Y_breeze - D_sub*X
      a = a+1
      if(a==nonZero) {
        //before exiting the loop, write the output out
        X_breeze = X
        
      }
      
    }
    
    
    var to_ret = BDM.zeros[Double](D_cols,1)
    //note: we can write the output to a sparse vector, however 
    //breeze provides csvwrite for dense matrices only
    //since in the final code, we wont need to write breeze vectors out to files
    //this shoudn't be an issue
    
    for(i<-0 to idx.length-1) {

     to_ret(idx(i),0) = X_breeze(i,0)
    }
    
    return to_ret
  }

  def main(args: Array[String]) {

  	val conf=new SparkConf().setAppName("omp")
    val sc=new SparkContext(conf)
    val Dcols = 6480
    val Drows = 1440
    val distFile=sc.textFile("Y.txt").flatMap(_.split(",").map(_.toDouble))
    var Y=Matrices.dense(1, Drows, distFile.collect())
    
    //read D
    val dValues = sc.textFile("D.txt").flatMap(_.split(",").map(_.toDouble))
    var D = new DenseMatrix(Dcols, Drows, dValues.collect()).transpose
    
    
    //perform OMP
    val nonZero = 10
    val X = OMP(Y, D, nonZero)
    //save output to file
    csvwrite(new File("scala_X.txt"), X, separator = ' ')
      
      sc.stop()
           
  }

}