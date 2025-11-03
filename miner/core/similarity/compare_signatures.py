import logging
from pyspark.sql import SparkSession


class CompareSignatures:

    def __init__(self, spark: SparkSession):
        self.logger = logging.getLogger(__name__)
        self.spark = spark

    # Assuming signatures of the same length
    def compare_signatures(self, sign1: list[int], sign2: list[int]) -> float:
        # Avoid vectors of length 0
        if not sign1 or not sign2:
            return 0.0
        matches = sum (a == b for a,b in zip(sign1,sign2))
        return matches/len(sign1)
    
    # Same function but done with Spark
    def compare_signatures_spark(self, sign1: list[int], sign2: list[int]) -> float:

        if not sign1:
            return 0.0
        
        sc = self.spark.sparkContext

        # First I attach to each element an index, then I swap the tuple to have the index first
        rdd1 = sc.parallelize(sign1).zipWithIndex().map(lambda v_idx: (v_idx[1],v_idx[0]))
        rdd2 = sc.parallelize(sign2).zipWithIndex().map(lambda v_idx: (v_idx[1],v_idx[0]))

        # Join based on index
        joined = rdd1.join(rdd2)
        
        # Calculate matches and total
        # N.B. we need the second lambda because Spark runs on partitions and we need to sum the result of each partition
        matches, total = joined.aggregate(
            (0, 0),
            lambda acc, x: (acc[0] + (1 if x[1][0] == x[1][1] else 0), acc[1] + 1),
            lambda a,b: (a[0]+b[0], a[1]+b[1])
        )

        # return similarity
        return float(matches) / float(total)
    


    
