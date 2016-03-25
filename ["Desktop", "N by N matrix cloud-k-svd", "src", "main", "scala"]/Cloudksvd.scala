/*
 * Licensed to the Paramathics Group(PG) under one or more
 * contributor license agreements.  
 */
package org.apache.spark.mllib.linalg.distributed.Cloudksvd

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import java.util.Arrays
import breeze.linalg.{axpy => brzAxpy, inv, svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg.{pinv, norm}
import breeze.numerics.{sqrt => brzSqrt}
import Math.sqrt

object Cloudksvd{
	/**
  * Read file from a String to Vector 
  * @param line is a String 
  */
  def readFile(line: String):Vector={
    var words=line.split(" ").map(x=>x.toDouble)
    Vectors.dense(words)
  }

  /**
  * Convert from a Matrix to Breeze Dense Matrix
  * @param v is Matrix
  */
  def toBreeze(v:Matrix):BDM[Double]={
    var m=v.numRows
    var n=v.numCols
    new BDM[Double](m,n,v.toArray)
  }

  /**
  * Convert from a Vector to Breeze Dense Vector
  * @param v is Vector
  */
  def toBreezeV(v:Vector): BDV[Double] = {
    v match {
      case DenseVector(values) =>
       new BDV[Double](values)
    }
  }

  /**
  * Convert from Breeze Dense Matrix to a Matrix
  * @param breeze is Breeze Dense Matrix
  */
  def fromBreeze(breeze: BDM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)

    }
  }

  /**
  * Convert from Breeze Dense Vecotr to a Vector
  * @param breezeVector is Breeze Dense Vector
  */
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

  /**
  * Convert from RDD[Vector] to a Breeze DenseMatrix
  * @param D is RDD[Vector]
  */
  def RDDtoBreeze(D:RDD[Vector]): BDM[Double] = {
    val m = D.count.toInt
    val n = D.first().size
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    D.collect().zipWithIndex.foreach { v =>
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

  /**
  * transpose RDD[Vector]
  * @param m is RDD[Vector]
  */
  def transposeRDD(m: RDD[Vector]): RDD[Vector] = {
    val transposedRowsRDD = m.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey().map(_._2) // sort rows and remove row indexes
      .map(buildRow) // restore order of elements in each row and remove column indexes
    transposedRowsRDD
  }

/**
  * calculate the norm value of a Breeze DenseMatrix
  * @param v is a breeze densematrix
  */
def normMatrix(v:BDM[Double]):Double={
  var num:Double=0;
  for(i<-0 until v.rows){
    for(j<- 0 until v.cols){
      num=num+v(i,j)*v(i,j)
    }
  }
  Math.sqrt(num)
}

/**
  * Normalize column of Breeze DenseMatrix
  * @param v is a breeze densematrix
  */
def normalizedCol(v:BDM[Double]):BDM[Double]={
  var mat=BDM.zeros[Double](v.rows,v.cols)
  for(i<-0 until v.cols){
    mat(::,i):=v(::,i)/norm(v(::,i))
  }
  mat
}

/**
  * Find and remove the zero vectors in Breeze DenseMatrix
  * @param D is a breeze densematrix
  */
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

/**
  * Using Orthogonal Matching pursuit to solve the Sparse Coding
  * @param Y is a breeze densematrix, the matrix you try to decomposite
  * @param D is a breeze densematrix, the Dictionary matrix
  * @param tol is the tolerance
  */
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


/**
  * Using Orthogonal Matching pursuit to solve the Sparse Coding
  * @param AT is the RDD[Vector], the matrix you tried to decomposite
  * @param D is a breeze densematrix, the Dictionary matrix
  * @param tol is the tolerance
  */
def SparseCoding_OMP(AT: RDD[Vector], D: BDM[Double], tol:Double):(BDM[Double],RDD[Vector])={
    var n=D.cols
    var Xrows=AT.zipWithIndex.map{case (a, index)=>
      var a_dm=toBreezeV(a).toDenseMatrix.t
      (index,SparseCoding(a_dm,D,tol))
    }

    var vector=Xrows.sortByKey().map(_._2.toDenseVector).map(x=>fromBreezeV(x))
    var Xcols_D=transposeRDD(vector).zipWithIndex.filter{case (v,index)=>
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

    var newD=Xcols_D.map{case (index, d, v)=>d}.reduce((x,y)=>x+y)
    (checkZero(newD),transposeRDD(Xcols))
}

/**
  * Generate a weight matrix for all nodes
  * @param n is the total number of nodes
  */
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


/**
  * Compute the Error matrix of target Matrix Y, and Dictionary matrix D and result matrix X
  * @param Y is the RDD[Vector], the target matrix Y, you tried to decomposite
  * @param D is the Breeze Dense Matrix, the dictionary matrix
  * @param X is the RDD[Vector], the result matrix X
  */
def computeCloudErr(Y:RDD[Vector] ,D:BDM[Double], X: RDD[Vector]):RDD[(Long,BDM[Double])]={
    var Yindex=Y.zipWithIndex.map{case (rows, rowIndex)=>(rowIndex,rows)};
    var Xindex=X.zipWithIndex.map{case (rows, rowIndex)=>(rowIndex,rows)};

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


/**
  * make all the cells in a densematrix multiply a number
  * @param A is the breeze densematrix
  * @param b is the number
  */
def dmMulnumber(A:BDM[Double],b:Double):BDM[Double]={
  var temp:BDM[Double]=BDM.zeros[Double](A.rows,A.cols)
  for(i<-0 until A.rows){
    for(j<-0 until A.cols){
      temp(i,j)=A(i,j)*b
    }
  }
  temp
}


/**
  * Cloud method by using consensus averaging method and power iteration method to update the dictionary
  * @param Y is the RDD[Vector], the target matrix Y, you tried to decomposite
  * @param D is the Breeze Dense Matrix, the dictionary matrix
  * @param X is the RDD[Vector], the result matrix X
  * @param Wt is the Breeze Dense Matrix, Weigth matrix of nodes
  * @param iterP is the power method iterations
  * @param iterC is the Consensus Averaging iterations
  */
def cloud(Y:RDD[Vector], D:BDM[Double], X: RDD[Vector], Wt:BDM[Double] ,iterP:Int, iterC:Int ):BDM[Double]={
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
}