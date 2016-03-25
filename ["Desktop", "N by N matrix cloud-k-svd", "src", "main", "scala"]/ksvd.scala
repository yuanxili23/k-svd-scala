import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.Cloudksvd._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}


object ksvd{
  def main(args: Array[String]) {
    val time_s:Double=System.nanoTime()

    val conf=new SparkConf().setMaster("local").setAppName("Cloud-KSVD")
          .set("spark.ui.port","4040")
    val sc=new SparkContext(conf)
    val distFile=sc.textFile("signal.txt").map(line => Cloudksvd.readFile(line))
    
    var AT=Cloudksvd.transposeRDD(distFile).cache()
    var n=AT.count.toInt // A is Y, which is k*n
    var k=distFile.count.toInt// D is  k*k
    // var W=GenW(n)

//--------gen W all connected----(for test)-----
    var W=BDM.zeros[Double](n,n)
    for(i<- 0 until n){
      for(j<- 0 until n){
        W(i,j)=1.0/n
      }
    }
//--------------------------------------

    var D=Cloudksvd.normalizedCol(BDM.rand(k,k))

    var t=args(0).toInt // for ksvd iteration
    var p=args(1).toInt // for power method iteration
    var c=args(2).toInt // for consensus averaging iteration
    var tol=args(3).toDouble

    var XT:RDD[Vector]=null

//--------- for printing res------------ 
    var X:BDM[Double]=null
    var resD:BDM[Double]=null
//--------------------------------------

    for(i<- 0 until t){
      var D_X=Cloudksvd.SparseCoding_OMP(AT,D,tol)
      D=D_X._1

//--------- for printing res------------      
      resD=D
//--------------------------------------

      XT=D_X._2
      
//-------------  print test-------------
      println("After SparseCoding")
      println("D: ")
      println(D)
      println("X: ")
      X=Cloudksvd.RDDtoBreeze(XT).t
      println(X)
      println("Y: ")
      println(D*X)
//--------------------------------------
      D=Cloudksvd.cloud(AT,D,XT,W,p,c)


//-------------  print test-------------
      println("After DicUpdate")
      println("D: ")
      println(D)
      println("X: ")
      println(X)
      println("Y: ")
      println(D*X)
//--------------------------------------
    }


//---------print running time-----------
    val time_e:Double=System.nanoTime()
    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")
//--------------------------------------
 

//--------  print error rate -----------
    println("Error rate:")
    var resY=resD*X
    var Y=Cloudksvd.RDDtoBreeze(distFile) //BDM
    println(Cloudksvd.normMatrix(resY-Y)/Cloudksvd.normMatrix(Y))
//--------------------------------------
    sc.stop()   
  }
}