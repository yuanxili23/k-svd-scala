import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import java.util.Arrays
import breeze.linalg.{axpy => brzAxpy, inv, svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV,
  MatrixSingularException, SparseVector => BSV}

import breeze.linalg.{pinv, norm}
import breeze.numerics.{sqrt => brzSqrt}
import Math.sqrt

object ksvd{

	def matrixToRDD(sc:SparkContext, m: Matrix): RDD[Vector] = {
   val columns = m.toArray.grouped(m.numRows)
   val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
   val vectors = rows.map(row => new DenseVector(row.toArray))
   sc.parallelize(vectors)
  }

  def readFile(line: String):Vector={
    var words=line.split(" ").map(x=>x.toDouble)
    Vectors.dense(words)
  }

  def toBreeze(v:Matrix):BDM[Double]={
    var m=v.numRows
    var n=v.numCols
    new BDM[Double](m,n,v.toArray)
  }

  def toBreezeV(v:Vector): BDV[Double] = {
    v match {
      case DenseVector(values) =>
       new BDV[Double](values)
    }
  }

  def fromBreeze(breeze: BDM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)

    }
  }

  def fromBreezeV(breezeVector: BDV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
    }
  }


  def RowMatrixtoBreeze(D:RowMatrix): BDM[Double] = {
    val m = D.numRows().toInt
    val n = D.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    D.rows.collect().zipWithIndex.foreach { v =>
      mat(v._2,::):=toBreezeV(v._1).t
    }
    mat
  }

  def rowToTransposedTriplet(row: Vector, rowIndex: Long): Array[(Long, (Long, Double))] = {
    val indexedRow = row.toArray.zipWithIndex
    indexedRow.map{case (value, colIndex) => (colIndex.toLong, (rowIndex, value))}
  }

  def buildRow(rowWithIndexes: Iterable[(Long, Double)]): Vector = {
    val resArr = new Array[Double](rowWithIndexes.size)
    rowWithIndexes.foreach{case (index, value) =>
        resArr(index.toInt) = value
    }
    Vectors.dense(resArr)
  } 


  def transposeRowMatrix(m: RowMatrix): RowMatrix = {
    val transposedRowsRDD = m.rows.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey().map(_._2) // sort rows and remove row indexes
      .map(buildRow) // restore order of elements in each row and remove column indexes
    new RowMatrix(transposedRowsRDD)
  }

def normMatrix(v:BDM[Double]):Double={
  var num:Double=0;
  for(i<-0 until v.rows){
    for(j<- 0 until v.cols){
      num=num+v(i,j)*v(i,j)
    }
  }
  Math.sqrt(num)
}

def normalizedCol(v:BDM[Double]):BDM[Double]={
  var mat=BDM.zeros[Double](v.rows,v.cols)
  for(i<-0 until v.cols){
    mat(::,i):=v(::,i)/norm(v(::,i))
  }
  mat
}

def checkZero(D:BDM[Double]): BDM[Double]={
  var m=D.cols
  var n=D.rows

  var checkZero=BDV.zeros[Double](n)
  var checklist:List[Int]=List()
    for(k<-0 until m){
      if(D(::,k)==checkZero){
        checklist=checklist:+k
      }
    }
    //use w remove 0
    checklist=checklist.sortWith(_>_)
    var len=checklist.length

    if(len==0){
      D
    }
    else{
        var Ddata=BDM.zeros[Double](D.rows,D.cols-len)
        for(i<- 0 until D.rows){
          var k=0
          for(j<- 0 until D.cols){
            if(!checklist.contains(j)){
              Ddata(i,k)=D(i,j)
              k=k+1
            }
          }
        }
      Ddata
    } 
}


def SparseCoding(Y:BDM[Double], D:BDM[Double], tol:Double):BDM[Double]={
    var n=Y.cols //for X matrix
    var m=D.cols
    var a:BDM[Double]=null
    var X=BDM.zeros[Double](m,1)
    var j=0
    while(j< n){
      var i=0
      var residue=Y
      var Dl=BDM.zeros[Double](D.rows,m)
      var selected_atom:List[Int]=List()
      var norm_R:Double=Double.PositiveInfinity
      while(i< m && norm_R> tol ){
          var r=residue(::,j)
          var index=0
          var maxValue=Double.NegativeInfinity
          for(k<- 0 until D.cols){
              var d=D(::,k)
              if(Math.abs(d.t*r)>maxValue){
                maxValue=Math.abs(d.t*r)
                index=k;
              }
          }
          selected_atom=selected_atom:+index
          Dl(::,i):=D(::,index)
          a=pinv(Dl)*Y
          residue=Y-Dl*a
          norm_R=normMatrix(residue)
          i=i+1         
      }

      selected_atom.zipWithIndex.foreach{case (v,k)=>
            X(v,j)=a(k,j)
      }
      j=j+1
    }
    X
}



def SparseCoding_OMP(AT: RowMatrix, D: BDM[Double], tol:Double):(BDM[Double],RowMatrix)={
    var n=D.cols
    var Xrows=AT.rows.zipWithIndex.map{case (a, index)=>
      var a_dm=toBreezeV(a).toDenseMatrix.t
      (index,SparseCoding(a_dm,D,tol))
    }

    var vector=Xrows.sortByKey().map(_._2.toDenseVector).map(x=>fromBreezeV(x))
    var Xcols_D=transposeRowMatrix(new RowMatrix(vector)).rows.zipWithIndex.filter{case (v,index)=>
      var dm_v=toBreezeV(v)
      var zeros=BDV.zeros[Double](dm_v.size)
      dm_v!=zeros
    }.map{case (v, index)=> 
      var newD=BDM.zeros[Double](D.rows,D.cols)
      newD(::,index.toInt):=D(::,index.toInt)
      (index,newD,v)
    }
    var Xcols=Xcols_D.map{case (index, d, v)=>
      (index,v)
    }.sortByKey().map(_._2)

    var Xrowmatrix=transposeRowMatrix(new RowMatrix(Xcols))

    var newD=Xcols_D.map{case (index, d, v)=>d}.reduce((x,y)=>x+y)
    (checkZero(newD),Xrowmatrix)
}


def GenW(n:Int):BDM[Double]={
  //gen upper matrix
  var a={
    var tmp=BDM.zeros[Double](n,n)
    var i=0
    var r=new scala.util.Random
    while(i< n){
      var j=i+1
      while(j< n){
        tmp(i,j)=r.nextInt(2)
        j=j+1
      }
      i=i+1
    }
    tmp
  }
  //gen full matrix
  var b={
    var i=0
    while(i< n){
      var j=i+1
      while(j< n){
        a(j,i)=a(i,j)
        j=j+1
      }
      i=i+1
    }
    a
  }
  var W=BDM.zeros[Double](n,n)
  var i=0
  while(i< n){
    var tmp1=b(i,::).t.toArray.filter(x=>x==1).length
    var j=0
    while(j< n){
      var tmp2=b(j,::).t.toArray.filter(x=>x==1).length
      if(b(i,j)!=0){
        W(i,j)=1.0/(Math.max(tmp1,tmp2)+1)
      }
      j+=1
    }
    i+=1
  }
  for(i<- 0 until n){
    var sum:Double=0
    for(j<- 0 until n){
      sum=sum+W(i,j)
    }
    W(i,i)=1-sum
  }
  W
}

def computeCloudErr(Y:RowMatrix ,D:BDM[Double], X: RowMatrix):RDD[(Long,BDM[Double])]={
    var Yindex=Y.rows.zipWithIndex.map{case (rows, rowIndex)=>(rowIndex,rows)};
    var Xindex=X.rows.zipWithIndex.map{case (rows, rowIndex)=>(rowIndex,rows)};

    var E_indiv=Yindex.join(Xindex).map{case (index, (row,xrow))=>
      var E=BDM.zeros[Double](D.rows,D.cols)
      var x=toBreezeV(xrow) 
        for(i<- 0 until D.cols){
          for(j<- 0 until D.cols){
            if(j!=i){
              E(::,i):=E(::,i)+D(::,j)*x(j)
            }
          }
        }
      var y=toBreezeV(row)
      for(i<-0 until D.cols){
        E(::,i):=y-E(::,i)
      }
      (index, E)
    }   
    E_indiv
}

def dmMulnumber(A:BDM[Double],b:Double):BDM[Double]={
  var temp:BDM[Double]=BDM.zeros[Double](A.rows,A.cols)
  for(i<-0 until A.rows){
    for(j<-0 until A.cols){
      temp(i,j)=A(i,j)*b
    }
  }
  temp
}

def cloud(Y:RowMatrix, D:BDM[Double], X: RowMatrix, Wt:BDM[Double] ,iterP:Int, iterC:Int ):BDM[Double]={
   var E=computeCloudErr(Y,D,X)
//initial Q-----------------------------------------------
   var Q=E.map{case (index,e)=>
    var q=normalizedCol(BDM.rand(D.rows,D.cols))
    (index, q)
   }
//--------------------------------------------------------
//for average consensus W computing  
   var W:BDM[Double]=Wt
   for(m<- 0 until iterC-1){    
      W=Wt*W
   }
//--------------------------------------------------------
   var E2=E.map{case (index, e)=>
        var res:Array[BDM[Double]]=Array()
        for(i<- 0 until e.cols){
          res=res:+e(::,i)*e(::,i).t
        }
        (index, res)
     }
//--------------------------------------------------------
    //power methods
  for(m<- 0 until iterP){  
   Q= E2.join(Q).map{case (index,(e,q))=>
        var Qpart=BDM.zeros[Double](D.rows,D.cols)
          for(i<- 0 until q.cols){
            var Z=e(i)*q(::,i)
            var V=Z/W(index.toInt,0)
            var new_q=V/sqrt(V.t*V)
            Qpart(::,i):=new_q
          }
      (index,Qpart)
    }.flatMap{case (index,q)=>
      var Seq= for(i<- 0 until W.cols) yield{
        (i,dmMulnumber(q,W(index.toInt,i)))
      }
      Seq
    }.reduceByKey((x,y)=>x+y).map{case (i,res)=>(i.toLong,res)}
  }
  Q.map(_._2).first()
}


  def main(args: Array[String]) {
    val time_s:Double=System.nanoTime()

    val conf=new SparkConf().setAppName("ksvd")
    val sc=new SparkContext(conf)
    val distFile=sc.textFile("signal.txt").map(line => readFile(line))
    

    var A=new RowMatrix(distFile)
    var AT=transposeRowMatrix(A)

    var n=AT.rows.count.toInt // A is Y, which is k*n
    var k=A.rows.count.toInt// D is  k*k
    // var W=GenW(n)

//gen W all connected-------------------
    var W=BDM.zeros[Double](n,n)

    for(i<- 0 until n){
      for(j<- 0 until n){
        W(i,j)=1.0/n
      }
    }
//--------------------------------------

    var D=normalizedCol(BDM.rand(k,k))

    var t=args(0).toInt // for ksvd iteration
    var p=args(1).toInt // for power method iteration
    var c=args(2).toInt // for consensus averaging iteration
    var tol=args(3).toDouble

    var XT:RowMatrix=null
    var X:BDM[Double]=null
    var resD:BDM[Double]=null

    for(i<- 0 until t){
      var D_X=SparseCoding_OMP(AT,D,tol)
      D=D_X._1
      resD=D

      XT=D_X._2
      
 
      println("After SparseCoding")
      println("D: ")
      println(D)
      println("X: ")
      X=RowMatrixtoBreeze(XT).t
      println(X)
      println("Y: ")
      println(D*X)

      D=cloud(AT,D,XT,W,p,c)

      println("After DicUpdate")
      println("D: ")
      println(D)
      println("X: ")
      println(X)
      println("Y: ")
      println(D*X)
    }

    val time_e:Double=System.nanoTime()
    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")

    

    println("Error rate:")
    var resY=resD*X
    var Y=RowMatrixtoBreeze(A) //BDM
    println(normMatrix(resY-Y)/normMatrix(Y))

    sc.stop()
    
  }
}