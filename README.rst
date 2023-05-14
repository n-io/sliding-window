
Sliding Window Tests
====================

This repo contains several sample programs building up primitives
for a sliding window FFT.

send-right-rpc
--------------
Sends ``num_elems`` wavelets over ``kernel_width`` PEs from left to right.
At the middle PE, the wavelets are sent down the router, and then back up
and further to the right.

sliding-window-rpc
------------------
Sends ``num_elems`` wavelets over ``kernel_width`` PEs from left to right,
and from right to left.
At the middle PE, the wavelets are sent down the router, and then back up,
heading further along the specified direction.
The initial values are copied into ``arr0``, and the flip-flopped values
will be in ``arr1``.

As an example:

.. code::

    // PE: 0 --- 1 --- 2 --- 3 --- 4 --- 5 --- 6 --- 7 --- 8

    // 0:  >     >     >     >     >
    //     ^                       v

    // 1:  <     <     <     <     <
    //     v                       ^

    // 2:                          <     <     <     <     <
    //                             v                       ^

    // 3:                          >     >     >     >     >
    //                             ^                       v

send-right-streaming and sliding-window-streaming
-------------------------------------------------
Versions of the above which use streaming instead of the RPC mechanism.
Not currently using or updating these.
