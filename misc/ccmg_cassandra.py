#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cassandra.cluster import Cluster, ExecutionProfile
from cassandra.policies import RetryPolicy

import time

# ===============================================================================


def retry_policy__on_read_timeout(query, consistency, required_responses,
                                  received_responses, data_retrieved,
                                  retry_num):
    if retry_num < 2:
        return (RetryPolicy.RETRY, None)
    return (RetryPolicy.RETHROW, None)


class cassandra:

    __cluster = None
    __session = None

    def __init__(self,
                 nodes=[
                     'r0b0.company', 'r0b1.company', 'r1b0.company',
                     'r1b1.company'
                 ],
                 keyspace='company_ng',
                 executor_threads=8,
                 request_timeout=10):
        self.__cluster = Cluster(nodes,
                                 connect_timeout=10,
                                 control_connection_timeout=10,
                                 executor_threads=executor_threads)
        self.__cluster.add_execution_profile(
            'default', ExecutionProfile(request_timeout=request_timeout))
        self.__cluster.default_retry_policy.on_read_timeout = retry_policy__on_read_timeout
        self.__session = self.__cluster.connect(keyspace)

    def session(self):
        return self.__session

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.__cluster.shutdown()
