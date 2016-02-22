import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition

import java.util.Arrays
import breeze.linalg.{axpy => brzAxpy, inv, svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV,
  MatrixSingularException, SparseVector => BSV}

import breeze.linalg.pinv
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import com.github.fommil.netlib.ARPACK
import org.netlib.util.{doubleW, intW}
import breeze.numerics.{sqrt => brzSqrt}



object ksvd{

	def matrixToRDD(sc:SparkContext, m: Matrix): RDD[Vector] = {
   val columns = m.toArray.grouped(m.numRows)
   val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
   val vectors = rows.map(row => new DenseVector(row.toArray))
   sc.parallelize(vectors)
  }


	def symmetricEigs(
      mul: BDV[Double] => BDV[Double],
      n: Int,
      k: Int,
      tol: Double,
      maxIterations: Int): (BDV[Double], BDM[Double]) = {
    // TODO: remove this function and use eigs in breeze when switching breeze version
    require(n > k, s"Number of required eigenvalues $k must be smaller than matrix dimension $n")

    val arpack = ARPACK.getInstance()

    // tolerance used in stopping criterion
    val tolW = new doubleW(tol)
    // number of desired eigenvalues, 0 < nev < n
    val nev = new intW(k)
    // nev Lanczos vectors are generated in the first iteration
    // ncv-nev Lanczos vectors are generated in each subsequent iteration
    // ncv must be smaller than n
    val ncv = math.min(2 * k, n)

    // "I" for standard eigenvalue problem, "G" for generalized eigenvalue problem
    val bmat = "I"
    // "LM" : compute the NEV largest (in magnitude) eigenvalues
    val which = "LM"

    var iparam = new Array[Int](11)
    // use exact shift in each iteration
    iparam(0) = 1
    // maximum number of Arnoldi update iterations, or the actual number of iterations on output
    iparam(2) = maxIterations
    // Mode 1: A*x = lambda*x, A symmetric
    iparam(6) = 1

    require(n * ncv.toLong <= Integer.MAX_VALUE && ncv * (ncv.toLong + 8) <= Integer.MAX_VALUE,
      s"k = $k and/or n = $n are too large to compute an eigendecomposition")

    var ido = new intW(0)
    var info = new intW(0)
    var resid = new Array[Double](n)
    var v = new Array[Double](n * ncv)
    var workd = new Array[Double](n * 3)
    var workl = new Array[Double](ncv * (ncv + 8))
    var ipntr = new Array[Int](11)

    // call ARPACK's reverse communication, first iteration with ido = 0
    arpack.dsaupd(ido, bmat, n, which, nev.`val`, tolW, resid, ncv, v, n, iparam, ipntr, workd,
      workl, workl.length, info)

    val w = BDV(workd)

    // ido = 99 : done flag in reverse communication
    while (ido.`val` != 99) {
      if (ido.`val` != -1 && ido.`val` != 1) {
        throw new IllegalStateException("ARPACK returns ido = " + ido.`val` +
            " This flag is not compatible with Mode 1: A*x = lambda*x, A symmetric.")
      }
      // multiply working vector with the matrix
      val inputOffset = ipntr(0) - 1
      val outputOffset = ipntr(1) - 1
      val x = w.slice(inputOffset, inputOffset + n)
      val y = w.slice(outputOffset, outputOffset + n)
      y := mul(x)
      // call ARPACK's reverse communication
      arpack.dsaupd(ido, bmat, n, which, nev.`val`, tolW, resid, ncv, v, n, iparam, ipntr,
        workd, workl, workl.length, info)
    }

    if (info.`val` != 0) {
      info.`val` match {
        case 1 => throw new IllegalStateException("ARPACK returns non-zero info = " + info.`val` +
            " Maximum number of iterations taken. (Refer ARPACK user guide for details)")
        case 3 => throw new IllegalStateException("ARPACK returns non-zero info = " + info.`val` +
            " No shifts could be applied. Try to increase NCV. " +
            "(Refer ARPACK user guide for details)")
        case _ => throw new IllegalStateException("ARPACK returns non-zero info = " + info.`val` +
            " Please refer ARPACK user guide for error message.")
      }
    }

    val d = new Array[Double](nev.`val`)
    val select = new Array[Boolean](ncv)
    // copy the Ritz vectors
    val z = java.util.Arrays.copyOfRange(v, 0, nev.`val` * n)

    // call ARPACK's post-processing for eigenvectors
    arpack.dseupd(true, "A", select, d, z, n, 0.0, bmat, n, which, nev, tol, resid, ncv, v, n,
      iparam, ipntr, workd, workl, workl.length, info)

    // number of computed eigenvalues, might be smaller than k
    val computed = iparam(4)

    val eigenPairs = java.util.Arrays.copyOfRange(d, 0, computed).zipWithIndex.map { r =>
      (r._1, java.util.Arrays.copyOfRange(z, r._2 * n, r._2 * n + n))
    }

    // sort the eigen-pairs in descending order
    val sortedEigenPairs = eigenPairs.sortBy(- _._1)

    // copy eigenvectors in descending order of eigenvalues
    val sortedU = BDM.zeros[Double](n, computed)
    sortedEigenPairs.zipWithIndex.foreach { r =>
      val b = r._2 * n
      var i = 0
      while (i < n) {
        sortedU.data(b + i) = r._1._2(i)
        i += 1
      }
    }

    (BDV[Double](sortedEigenPairs.map(_._1)), sortedU)
  }

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

  def SparseCoding_OMP(rows: RDD[Vector], D: BDM[Double], L:Int): (Matrix, Matrix) ={
    var m=D.rows
    var n=D.cols
    var matrix=new RowMatrix(rows)
    var Y=RowMatrixtoBreeze(matrix)
    var newX=BDM.zeros[Double](n,n)
    var residue=Y

    for(i<- 0 until L){
      var dot_p=D.t*residue
      for(i<-0 until n){
         for(j<-0 until n){
            dot_p(i,j)=Math.abs(dot_p(i,j))
         }
      }
      var newD=BDM.zeros[Double](m,n)
      var index=dot_p.argmax
      var j=0;
      index.productIterator.map(_.asInstanceOf[Int]).foreach{i=>
        newD(::,j):=D(::,i)
        j=j+1
      }
      newX=pinv(newD)*Y
      residue=Y-newD*newX
    }

    (fromBreeze(D),fromBreeze(newX))
  }


  def spr(alpha: Double, v: Vector, U: Array[Double]): Unit = {
    val n = v.size
    v match {
      case DenseVector(values) =>
        NativeBLAS.dspr("U", n, alpha, values, 1, U)
    }
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

  // def transposeRDDVector(m:RDD[Vector]):RowMatrix={
  //   val transposedRowsRDD = m.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
  //     .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
  //     .groupByKey
  //     .sortByKey().map(_._2) // sort rows and remove row indexes
  //     .map(buildRow) // restore order of elements in each row and remove column indexes
  //   new RowMatrix(transposedRowsRDD)

  // }


  

  
  def constructRow(rowWithIndexes: (Long, Double)):Vector ={
    var res=Array.ofDim[Double](1)
    
    val value=rowWithIndexes._2
    res(0)=value
    Vectors.dense(res)
  }


  def getNthcols(rowIndex: RDD[(Vector,Long)]):RowMatrix = {
    val transposedRowsRDD = rowIndex.map{case (row, rowIndex) => (rowIndex, row(0))}
    .sortByKey().map(constructRow)
    new RowMatrix(transposedRowsRDD)
  }

  



  def triuToFull(n: Int, U: Array[Double]): Matrix = {
    val G = new BDM[Double](n, n)

    var row = 0
    var col = 0
    var idx = 0
    var value = 0.0
    while (col < n) {
      row = 0
      while (row < col) {
        value = U(idx)
        G(row, col) = value
        G(col, row) = value
        idx += 1
        row += 1
      }
      G(col, col) = U(idx)
      idx += 1
      col +=1
    }

    Matrices.dense(n, n, G.data)
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



  
  def computeGramian(v: Matrix):Matrix={
  	var m=v.numRows
  	var n=v.numCols
  	var localv=new BDM[Double](m,n,v.toArray)
  	var ATA=localv.t*localv
  	Matrices.dense(n,n, ATA.toArray)
  }
  
  def multiplyGramianMatrixBy(D:RowMatrix, v: BDV[Double]): BDV[Double] = {
    var n=D.numCols().toInt
    val vbr = D.rows.context.broadcast(v)
    D.rows.treeAggregate(BDV.zeros[Double](n))(
      seqOp = (U, r) => {
        val rBrz = toBreezeV(r)
        val a = rBrz.dot(vbr.value)
        rBrz match {
          // use specialized axpy for better performance
          case _: BDV[_] => brzAxpy(a, rBrz.asInstanceOf[BDV[Double]], U)
        }
        U
      }, combOp = (U1, U2) => U1 += U2)
  }


  def computeSVD(D:RowMatrix ,G: BDM[Double], k:Int):(Double, BDM[Double])={
     val tol = 1e-10
     val maxIter=math.max(300,k*3)
     val n=G.cols
     object SVDMode extends Enumeration {
      val LocalARPACK, LocalLAPACK, DistARPACK = Value
     }
     var mode="auto"
     val computeMode = mode match {
      case "auto" =>
        // TODO: The conditions below are not fully tested.
        if (n < 100 || (k > n / 2 && n <= 15000)) {
          // If n is small or k is large compared with n, we better compute the Gramian matrix first
          // and then compute its eigenvalues locally, instead of making multiple passes.
          if (k < n / 3) {
            SVDMode.LocalARPACK
          } else {
            SVDMode.LocalLAPACK
          }
        } else {
          // If k is small compared with n, we use ARPACK with distributed multiplication.
          SVDMode.DistARPACK
        }
      case "local-svd" => SVDMode.LocalLAPACK
      case "local-eigs" => SVDMode.LocalARPACK
      // case "dist-eigs" => SVDMode.DistARPACK
      case _ => throw new IllegalArgumentException(s"Do not support mode $mode.")
    }
    val (sigmaSquares: BDV[Double], u: BDM[Double]) = computeMode match {
      case SVDMode.LocalARPACK =>
        symmetricEigs(v => G * v, n, k, tol, maxIter)
      case SVDMode.LocalLAPACK =>
        val brzSvd.SVD(uFull: BDM[Double], sigmaSquaresFull: BDV[Double], _) = brzSvd(G)
        (sigmaSquaresFull, uFull)
      // case SVDMode.DistARPACK =>
      //  symmetricEigs(v=> multiplyGramianMatrixBy(D,v) , n, k, tol, maxIter)
    }
     // var (sigmaSquares: BDV[Double], u: BDM[Double])=symmetricEigs(v => G * v, n, k, tol, maxIter)
     val sigmas: BDV[Double] = brzSqrt(sigmaSquares)
     val sigma0 = sigmas(0)
     (sigma0, u) 
  }


  


  def computeSigmaAndV(D:RowMatrix, X: Matrix):RDD[(Long,(Double,BDM[Double]))] ={
  	var DT=transposeRowMatrix(D)

  	var Drow=D.numRows()
  	var Dcol=D.numCols()
  	// var Dcol=getNthcols(DwithIndex)
  	var Xarray=DT.rows.zipWithIndex.map{case (rows, rowIndex)=>
  			(rowIndex,X.toArray.zipWithIndex.filter{case (value, index)=>
		  		index%2==rowIndex
		  	}.map(_._1))
  		}  

  	var E=DT.rows.zipWithIndex.map{case (rows, rowIndex)=> (rowIndex, rows.toArray)}.join(Xarray).map{case (index, (x,y))=>
  			(index, Matrices.dense(Drow.toInt,Dcol.toInt,(BDV(x)*BDV(y).t).toArray))
  		}
  	var G=E.map{case (index, v)=> (index, toBreeze(computeGramian(v)))}
  	var svd=G.map{case (i, grammian) => (i, computeSVD(D, grammian,1))}.map{case (i, (sigma, ufull))=> 
        (i, (sigma,ufull(::,0).toDenseMatrix)) }
    svd
  }


  def updateX(svd:RDD[(Long, (Double, BDM[Double]))], X:Matrix):Matrix={
     var cols=X.numCols
     var BDM_X=svd.sortByKey().map{ x=> x._2._1*x._2._2.t}.reduce((x,y)=>BDM.vertcat(x.reshape(1,cols),y.reshape(1,cols)))
     fromBreeze(BDM_X)
  }




  def computeU(sc:SparkContext, E:RDD[(Long,Matrix)], vd:RDD[(Long,(Double,BDM[Double]))]):BDM[Double]={
    var E_v=E.join(vd).map{case (index, (e,(sigma, v))) =>
      (index, toBreeze(e), v) 
      }.map{case (index, e, v)=> (index,e*v.t)}.sortByKey().map(_._2)
    var D=E_v.reduce((x,y)=>BDM.horzcat(x,y))
    D
  }

  
  def computeErr(D:RowMatrix, X: Matrix):RDD[(Long,Matrix)]={
    var DT=transposeRowMatrix(D)

    var Drow=D.numRows()
    var Dcol=D.numCols()
    // var Dcol=getNthcols(DwithIndex)
    var Xarray=DT.rows.zipWithIndex.map{case (rows, rowIndex)=>
        (rowIndex,X.toArray.zipWithIndex.filter{case (value, index)=>
          index%2==rowIndex
        }.map(_._1))
      }  

    var E=DT.rows.zipWithIndex.map{case (rows, rowIndex)=> (rowIndex, rows.toArray)}.join(Xarray).map{case (index, (x,y))=>
        (index, Matrices.dense(Drow.toInt,Dcol.toInt,(BDV(x)*BDV(y).t).toArray))
      }.sortByKey()
    E
  }



  def main(args: Array[String]) {
    
    val conf=new SparkConf().setAppName("ksvd")
    val sc=new SparkContext(conf)
    val distFile=sc.textFile("/Users/Yuanxi/Desktop/distri-k-svd/signal.txt").map(line => readFile(line))
    var A=new RowMatrix(distFile)

    var n=A.rows.count.toInt
    var k=2 // D is  n*k
    var D=BDM.rand(n,k)
    //var qrResult=SparseCoding(distFile)
    var L=2
    var t=args(0).toInt
    //L: number of non-zero entries in output

    for(i<- 0 until t){
      var qrResult=SparseCoding_OMP(distFile,D,L)

      var Drowmatrix=new RowMatrix(matrixToRDD(sc, qrResult._1))
      var X=qrResult._2

      var vd=computeSigmaAndV(Drowmatrix,X)

      var E=computeErr(Drowmatrix,X)
      D=computeU(sc, E, vd)
      X=updateX(vd,X)
    }
    // var res=DicUpdate(D,X)
    // var DT=transposeRowMatrix(D)
    // DT.rows.foreach(println)
    // D.rows.foreach(println)
    // println(X)
    // res.rows.foreach(println)
    println("D:")
    D.rows.foreach(println)
    println("X:")
    println(X)
    var Y=D.multiply(X)
    println("Y:")
    Y.rows.foreach(println)
    

    

    sc.stop()
    
  }
}