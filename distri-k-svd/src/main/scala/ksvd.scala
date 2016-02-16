import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition

import breeze.linalg.{svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}


object ksvd{
  def readFile(line: String):Vector={
  	var words=line.split(",").map(x=>x.toDouble)
  	Vectors.dense(words)
  }

  def SparseCoding(rows: RDD[Vector]): (RowMatrix, Matrix) ={
  	var mat:RowMatrix = new RowMatrix(rows)
    var qrResult=mat.tallSkinnyQR(computeQ=true)
    var Q=qrResult.Q
    var R=qrResult.R

    (Q,R)
  }

  def transposeRowMatrix(m: RowMatrix): RowMatrix = {
    val transposedRowsRDD = m.rows.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey().map(_._2) // sort rows and remove row indexes
      .map(buildRow) // restore order of elements in each row and remove column indexes
    new RowMatrix(transposedRowsRDD)
  }


  def rowToTransposedTriplet(row: Vector, rowIndex: Long): Array[(Long, (Long, Double))] = {
    val indexedRow = row.toArray.zipWithIndex
    indexedRow.map{case (value, colIndex) => (colIndex.toLong, (rowIndex, value))}
  }

  def getNthcols(rowIndex: RDD[(Vector,Long)]):RowMatrix = {
    val transposedRowsRDD = rowIndex.map{case (row, rowIndex) => (rowIndex, row(0))}
    .sortByKey().map(constructRow)
    new RowMatrix(transposedRowsRDD)
  }

  def constructRow(rowWithIndexes: (Long, Double)):Vector ={
    var res=Array.ofDim[Double](1)
    
    val value=rowWithIndexes._2
    res(0)=value
    Vectors.dense(res)
  }

  def buildRow(rowWithIndexes: Iterable[(Long, Double)]): Vector = {
    val resArr = new Array[Double](rowWithIndexes.size)
    rowWithIndexes.foreach{case (index, value) =>
        resArr(index.toInt) = value
    }
    Vectors.dense(resArr)
  } 


  def DicUpdate(D:RowMatrix, X: Matrix):RowMatrix ={
  	var Dcols=D.numCols().toInt
  	var Xcols=X.numCols

  	var DT=transposeRowMatrix(D)

  	// var Dcol=getNthcols(DwithIndex)
  	var Xarray=DT.rows.zipWithIndex.map{case (rows, rowIndex)=>
  			(rowIndex,X.toArray.zipWithIndex.filter{case (value, index)=>
		  		index%2==rowIndex
		  	}.map(_._1))
  		}  
  	var E=DT.rows.zipWithIndex.map{case (rows, rowIndex)=> (rowIndex, rows.toArray)}.join(Xarray)
  		.map{case (index, (Dcol,Xrow)=>
  			
  		}
  	// var E=Dcol.multiply(denseX)
  }


  def main(args: Array[String]) {
    
    val conf=new SparkConf().setAppName("ksvd")
    val sc=new SparkContext(conf)
    val distFile=sc.textFile("signal.txt").map(line => readFile(line))
    var qrResult=SparseCoding(distFile)
    var D=qrResult._1
    var X=qrResult._2
    // var res=DicUpdate(D,X)

    D.rows.foreach(println)
    println(X)
    res.rows.foreach(println)

    

    

    sc.stop()
    
  }
}