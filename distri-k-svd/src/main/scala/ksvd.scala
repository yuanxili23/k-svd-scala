import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition

import java.util.Arrays
import breeze.linalg.{axpy => brzAxpy, inv, svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV,
  MatrixSingularException, SparseVector => BSV}

import breeze.linalg.{pinv, norm}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import com.github.fommil.netlib.ARPACK
import org.netlib.util.{doubleW, intW}
import breeze.numerics.{sqrt => brzSqrt}
import Math.sqrt

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


  def computeSVD(G: BDM[Double], k:Int):(Double, BDM[Double])={
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


  


  def computeSigmaAndV(Y:BDM[Double], D:RowMatrix, X: RowMatrix):RDD[(Long,(Vector,(Double,BDM[Double])))] ={
  var DT=transposeRowMatrix(D)
  var Drow=D.numRows()
  var Dcol=D.numCols()
  // var Dcol=getNthcols(DwithIndex)
  var Xarray=X.rows.zipWithIndex.map{case (rows, rowIndex)=>(rowIndex,rows)};
  var DTindex=DT.rows.zipWithIndex.map{case (rows, rowIndex)=> (rowIndex, rows)}
  var E_indiv=DTindex.join(Xarray).map{case (index, (x,y))=>(index,toBreezeV(x)*toBreezeV(y).t)}
  var Eall=Y-E_indiv.map(_._2).reduce((x,y)=>x+y)
  var E=E_indiv.map{case (index, v)=> (index, fromBreeze(Eall+v) ) }

  var G=E.map{case (index, v)=> (index, toBreeze(computeGramian(v)))}
  var svd=DTindex.join(G).map{case (i, (d, grammian)) => (i, (d, computeSVD(grammian,1)))}
  svd
  }


  // def updateX (svd:RDD[(Long, (Double, BDM[Double]))], X:Matrix): Matrix={
  //    var cols=X.numCols
  //    var BDM_X=svd.sortByKey().map{ x=> x._2._1*x._2._2.t}.reduce((x,y)=>BDM.vertcat(x.reshape(1,cols),y.reshape(1,cols)))
  //    fromBreeze(BDM_X)
  // }




  def computeU(E:RDD[(Long,Matrix)], vd:RDD[(Long,(Vector, (Double,BDM[Double])))]):BDM[Double]={
    var rows=E.take(1)(0)._2.numRows
    var cols=E.count.toInt
    var D=BDM.zeros[Double](rows,cols)
    var E_v=E.join(vd).map{case (index, (e, (d, (sigma, v)))) =>
      (index, d, toBreeze(e)*v.t/sigma)
    }.map{ case (index, d, v) => 
      if(d(0)*v(0,0)< 0){
        (index, -v)
      }
      else{
        (index, v)
      }
    }
    E_v.collect().foreach(x=> D(::,x._1.toInt):=x._2.toDenseVector)
    D
  }

  
  def computeErr(Y:BDM[Double] ,D:RowMatrix, X: RowMatrix):RDD[(Long,Matrix)]={
    var DT=transposeRowMatrix(D)

    var Drow=D.numRows()
    var Dcol=D.numCols()
    // var Dcol=getNthcols(DwithIndex)
    var Xarray=X.rows.zipWithIndex.map{case (rows, rowIndex)=>(rowIndex,rows)};

    var DTindex=DT.rows.zipWithIndex.map{case (rows, rowIndex)=> (rowIndex, rows)}

    var E_indiv=DTindex.join(Xarray).map{case (index, (x,y))=>(index,toBreezeV(x)*toBreezeV(y).t)}
    var Eall=Y-E_indiv.map(_._2).reduce((x,y)=>x+y)
    var E=E_indiv.map{case (index, v)=> (index, fromBreeze(Eall+v) ) }

    E
  }


  def SparseCoding_BMP(A: RowMatrix, D: BDM[Double], L:Int): (Matrix,Matrix) ={
    var rows=A.rows

    var m=D.cols
    var n=A.numCols().toInt //for X matrix

    var matrix=new RowMatrix(rows)
    var Y=RowMatrixtoBreeze(matrix)
    var X=BDM.zeros[Double](m,n)
    var residue=Y

    for(i<- 0 until L){
      for(j<- 0 until residue.cols ){
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
        var value=D(::,index).t*r
        X(index,j)=X(index,j)+value
        residue(::,j):=residue(::,j)-value*D(::,index)   
      }
    }
    (fromBreeze(D), fromBreeze(X))
  }

  def SparseCoding_OMP(A: RowMatrix, D: BDM[Double], tol:Double): (Matrix,Matrix) ={
    var rows=A.rows
    var m=D.cols
    var Dl=BDM.zeros[Double](D.rows,m)
    var n=A.numCols().toInt //for X matrix
    // var matrix=new RowMatrix(rows)
    var Y=RowMatrixtoBreeze(A)
    var a=BDM.zeros[Double](m,n)
    var X=BDM.zeros[Double](m,n)
    //initial selected_atom
    var selected_atom:List[List[Int]]=List()
    for(i<- 0 until n){
      selected_atom=selected_atom:+List()
    }
    var residue=Y
    var i=0

    while(i< m){
      var j=0
      while (j< residue.cols && normMatrix(residue) > tol ){
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

        selected_atom=selected_atom.updated(j, selected_atom(j):+index)
        Dl(::,i):=D(::,index)
        a=pinv(Dl)*Y
        residue=Y-Dl*a
        j=j+1
        print("residue is ")
        println(normMatrix(residue))  
      }
      i=i+1

      selected_atom.zipWithIndex.foreach{case (v1,i)=>
        v1.zipWithIndex.foreach{case (v2, j)=>
           X(v2,i)=a(j,i)
        }
      }

    }
    checkZero(D,X)
}

def checkZero(D:BDM[Double],X:BDM[Double]): (Matrix,Matrix)={
  var m=D.cols
  var n=X.cols

  var checkZero=BDV.zeros[Double](n)
    var checklist:List[Int]=List()
    for(k<-0 until m){
      if(X(k,::).t==checkZero){
        checklist=checklist:+k
      }
    }
    //use w remove 0
    checklist=checklist.sortWith(_>_)
    var len=checklist.length

    if(len==0){
      (fromBreeze(D),fromBreeze(X))
    }
    else{
      var Ddata=D.data
      var Xdata=X.data
      // println(checklist)
        checklist.foreach{i=>
          Ddata=Ddata.zipWithIndex.filter(x=>x._2< i*D.rows || x._2 >=(i+1)*D.rows ).map(_._1)
          Xdata=Xdata.zipWithIndex.filter(x=>x._2< i*X.cols || x._2 >=(i+1)*X.cols ).map(_._1)
        }
      (new DenseMatrix(D.rows,D.cols-len, Ddata), new DenseMatrix(X.rows-len, X.cols, Xdata) ) 
    } 
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

def distributed(Y:BDM[Double] ,D:RowMatrix, X: RowMatrix, Q:RowMatrix, W: BDM[Double], iter:Int):BDM[Double]={
   var Xrows=X.numRows.toInt
   var Xcols=X.numCols.toInt
   var E2=computeErr(Y,D,X).map{case (index, v) => (index, toBreeze(v)*toBreeze(v.transpose) ) }
   var Qindex=Q.rows.zipWithIndex.map{case (row, index)=> (index, row) }
   
   var updatedD=BDM.zeros[Double](D.numRows.toInt, D.numCols.toInt)

   for(m<- 0 until iter){
     var Z=E2.join(Qindex).map{case (index, (v,q))=>(index, v*toBreezeV(q))}.map{case(index, z)=> (index, W*z)}

     var V=Z.map{case (index,z)=> (index, z/W(0,index.toInt)) }
     Qindex=V.map{case (index,v)=> 
       var r=v.t*v
       var q=v/r
      (index, fromBreezeV(q))
     }
   }

   Qindex.collect().foreach{case (index, v)=>
      updatedD(::,index.toInt):=toBreezeV(v)
   }

   updatedD
}


  def main(args: Array[String]) {
    
    val conf=new SparkConf().setAppName("ksvd")
    val sc=new SparkContext(conf)
    val distFile=sc.textFile("/Users/Yuanxi/Desktop/distri-k-svd/signal.txt").map(line => readFile(line))
    
    var A=new RowMatrix(distFile)
    var Y=RowMatrixtoBreeze(A) //BDM

    var n=A.rows.count.toInt
    var k=n// D is  n*k

    var D=normalizedCol(BDM.rand(n,k))

    //var qrResult=SparseCoding(distFile)
    var tol:Double=1 // for sparse coding iteration

    var t=args(0).toInt // for ksvd iteration


    var Xrows=n

      //create weight matrix
    var W=BDM.rand(Xrows,Xrows)
      //initial Q
    var Q=new RowMatrix(matrixToRDD(sc, fromBreeze(normalizedCol(BDM.rand(Xrows,Xrows)).t)))


    for(i<- 0 until t){
      var qrResult=SparseCoding_OMP(A,D,tol)

      var Drowmatrix=new RowMatrix(matrixToRDD(sc, qrResult._1))
      var Xrowmatrix=new RowMatrix(matrixToRDD(sc, qrResult._2))


      
 
      println("After SparseCoding")
      println("D: ")
      println(toBreeze(qrResult._1))
      println("X: ")
      println(toBreeze(qrResult._2))
      println("Y: ")
      println(toBreeze(qrResult._1)*toBreeze(qrResult._2))

//method 1
      var vd=computeSigmaAndV(Y,Drowmatrix,Xrowmatrix)
      var E=computeErr(Y,Drowmatrix,Xrowmatrix)
      D=computeU(E, vd)

      println("After DicUpdate")
      println("D: ")
      println(D)
      println("X: ")
      println(toBreeze(qrResult._2))
      println("Y: ")
      println(D*toBreeze(qrResult._2))
//method2
      // D=distributed(Y,Drowmatrix,Xrowmatrix,Q,W,3)

      // println("After DicUpdate")
      // println("D: ")
      // println(D)
      // println("X: ")
      // println(toBreeze(qrResult._2))
      // println("Y: ")
      // println(D*toBreeze(qrResult._2))


    }

    // var res=DicUpdate(D,X)
    // var DT=transposeRowMatrix(D)
    // DT.rows.foreach(println)
    // D.rows.foreach(println)
    // println(X)
    // res.rows.foreach(println)
    // var X_BDM=toBreeze(X)
    // println("D:")
    // println(D)
    // println("X:")
    // println(X_BDM)
    // var Y=D*X_BDM
    // println("Y:")
    // println(Y)    

    

    sc.stop()
    
  }
}