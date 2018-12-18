import java.io.{File, PrintWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import scala.collection.immutable.ListMap
import scala.collection.mutable.HashMap

object Shivani_Shukla_ModelBasedCF {
  val output_file = "Shivani_Shukla_ModelBasedCF.txt"

  def compute_range(value: Double): Int = {
    if (value >= 0.0 && value < 1.0) {
      0
    } else if(value >= 1.0 && value < 2.0) {
      1
    } else if(value >= 2.0 && value < 3.0) {
      2
    } else if(value >= 3.0 && value < 4.0) {
      3
    } else {
      4
    }
  }

  def main(args: Array[String]): Unit = {

    val t0 = System.currentTimeMillis()
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("Model Based CF").setMaster("local[1]")
    val sc = new SparkContext(conf)

    var count1 = 0
    var count2 = 0

    var users_map = new scala.collection.mutable.HashMap[String, Int]
    var businesses_map = new scala.collection.mutable.HashMap[String, Int]

    var users_map_print = new scala.collection.mutable.HashMap[Int, String]
    var businesses_map_print = new scala.collection.mutable.HashMap[Int, String]

    val csvRDD = sc.textFile(args(0))
    //csvRDD.take(10).foreach(println)
    val header = csvRDD.first()

    val testData = sc.textFile(args(1))

    val csvRDDwithoutHeader = csvRDD.filter(_ != header )
    //csvRDDwithoutHeader.take(10).foreach(println)
    val testRDDwithoutHeader = testData.filter(_ != header )


    //Train RDD
    val train_RDD1 = csvRDDwithoutHeader.map(line =>{
      val data = line.split(",")
      (data(0),data(1),data(2))
    })

    //Test RDD
    val test_RDD1 = testRDDwithoutHeader.map(line =>{
      val data = line.split(",")
      (data(0),data(1),data(2))
    })

    //Train HashMaps
    train_RDD1.collect().foreach(X => {
      if(!users_map.contains(X._1)) {
        //users_map_print(count1) = X._1
        users_map.put(X._1, count1)
        users_map_print.put(count1, X._1)
        count1 = count1 + 1
      }

      if(!businesses_map.contains(X._2)){
        //businesses_map_print(count2) = X._2
        businesses_map.put(X._2, count2)
        businesses_map_print.put(count2, X._2)
        count2 = count2 +  1
      }

    })

    //Test HashMaps
    test_RDD1.collect().foreach(X => {
      if(!users_map.contains(X._1)) {
        //users_map_print(count1) = X._1
        users_map.put(X._1, count1)
        users_map_print.put(count1, X._1)
        count1 = count1 + 1
      }

      if(!businesses_map.contains(X._2)){
        //businesses_map_print(count2) = X._2
        businesses_map.put(X._2, count2)
        businesses_map_print.put(count2, X._2)
        count2 = count2 +  1
      }

    })

    val ratings = train_RDD1.map(line => {
      Rating(users_map(line._1), businesses_map(line._2), line._3.toDouble)
    })

    // Build the recommendation model using ALS
    val rank = 2
    val numIterations = 22
    val lambda = 0.3
    val seed = 2
    val blocks = -1


    val model = ALS.train(
      ratings, rank, numIterations, lambda, seed, blocks
    )


    // Evaluate the model on rating data
    val testMap = test_RDD1.map(line => {
      Rating(users_map(line._1), businesses_map(line._2), line._3.toDouble)
    })

    val usersBusinesses = testMap.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val predictions =
      model.predict(usersBusinesses).map { case Rating(user, product, rate) => ((user, product), rate)
      }

    val ratesAndPreds = testMap.map { case Rating(user, product, rate) =>
          ((user, product), rate)
        }.join(predictions)


    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
          val err = (r1 - r2)
          err * err
        }.mean()

    val RMSE = Math.sqrt(MSE)
    //println(RMSE)

    //baseline = ListMap(baseline.toSeq.sortBy(_._1):_*)
    var baseline = ratesAndPreds.map{ case ((user, product), (r1, r2)) =>
      (compute_range(math.abs(r1 - r2)),1)}.countByKey()

    baseline = ListMap(baseline.toSeq.sortBy(_._1):_*)


    val predictions_dict = predictions.collect()
    val writer = new PrintWriter(new File(output_file))
    val last_index = predictions_dict.size - 1

    for (((user_business, r), index) <- predictions_dict.zipWithIndex) {

      val formatted_string = Array(
        users_map_print(user_business._1),
        businesses_map_print(user_business._2),
        r
      ).mkString(", ")

      if (index != last_index) {
        writer.write(s"$formatted_string\n")
      } else {
        writer.write(s"$formatted_string")
      }
    }
    writer.close()

    for((k,v)<-baseline) {
      if(k == 0) {
        println(">=0 and <1: "+ v)
      } else if (k == 1) {
        println(">=1 and <2: "+ v)
      } else if(k == 2) {
        println(">=2 and <3: "+ v)
      } else if(k == 3) {
        println(">=3 and <4: "+ v)
      } else {
        println(">=4: "+ v)
      }
    }
    //    println(baseline)

    println("RMSE: "+ RMSE)
    println("Time: " + (System.currentTimeMillis() - t0)/1000 + "sec")

}}


