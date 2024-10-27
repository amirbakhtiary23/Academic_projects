// Library:  calc3
// Module:  Priority logic
// Author:  Naseer Siddique

module priority (port1_invalid_op, port1_invalid_tag,  port2_invalid_op, port2_invalid_tag, port3_invalid_op, port3_invalid_tag, port4_invalid_op, port4_invalid_tag, prio_adder_cmd, prio_adder_data1, prio_adder_data2, prio_adder_follow_branch, prio_adder_out_vld, prio_adder_result, prio_adder_tag, prio_shift_cmd, prio_shift_data1, prio_shift_data2, prio_shift_data, prio_shift_follow_branch, prio_shift_out_vld,  prio_shift_result, prio_shift_tag, scan_out, a_clk, b_clk, c_clk, hold1_cmd, hold1_data1, hold1_data2, hold1_data,  hold1_result, hold1_tag, hold2_cmd,hold2_data1, hold2_data2, hold2_data, hold2_result, hold2_tag, hold3_cmd, hold3_data1, hold3_data2,hold3_data,hold3_result, hold3_tag, hold4_cmd , hold4_data1, hold4_data2, hold4_data, hold4_result, hold4_tag, reset, scan_in);

   output port1_invalid_op, port2_invalid_op, port3_invalid_op, port4_invalid_op, prio_adder_out_vld,prio_shift_out_vld, scan_out;
   output [0:1] port1_invalid_tag, port2_invalid_tag, port3_invalid_tag, port4_invalid_tag;
   output [0:3] prio_adder_cmd, prio_adder_tag, prio_shift_cmd, prio_shift_tag;
   output [0:4] prio_adder_data1, prio_adder_data2, prio_adder_follow_branch, prio_adder_result, prio_shift_data1, prio_shift_data2, prio_shift_follow_branch, prio_shift_result;
   output [0:31] prio_shift_data;

   input 	 a_clk, b_clk, c_clk, reset, scan_in;
   input [0:3] 	 hold1_cmd, hold1_data1, hold1_data2, hold1_result, hold2_cmd, hold2_data1, hold2_data2, hold2_result, hold3_cmd, hold3_data1, hold3_data2, hold3_result, hold4_cmd, hold4_data1, hold4_data2, hold4_result;
   input [0:31]  hold1_data, hold2_data, hold3_data, hold4_data;
   input [0:1] 	 hold1_tag, hold2_tag, hold3_tag, hold4_tag;
   
   wire [0:4] 	 hold1_add_pos, hold2_add_pos, hold3_add_pos, hold4_add_pos;
   wire [0:4] 	 hold1_shift_pos,  hold2_shift_pos, hold3_shift_pos, hold4_shift_pos;
   wire [0:4] 	 hold1_add_valid, hold2_add_valid, hold3_add_valid, hold4_add_valid;
   wire [0:4] 	 hold1_shift_valid, hold2_shift_valid, hold3_shift_valid, hold4_shift_valid;
   reg [0:2] 	 hold1_invalid, hold2_invalid, hold3_invalid, hold4_invalid;
   reg [0:4] 	 addqueue_curpos;
   reg [0:4] 	 shiftqueue_curpos;

   wire [0:4] 	 temp_dispatch_add, temp_dispatch_shift, dispatch_add, dispatch_shift, rd_wr_conflict_block1, rd_wr_conflict_block2;
   wire 	 temp_hazard, hazard1, hazard2, rd_wr_conflict, skip_11, add_and_shift, block_condition;
   
   wire [0:2] 	 a, b;
   wire [0:61] 	 c1a, c2a, c1b, c2b, c1a2, c2a2,c1b2, c2b2;
   
   
   //   reg [0:3] 	 add_queue0, add_queue1, add_queue2, add_queue3, add_queue4, add_queue5, add_queue6,  add_queue7,  add_queue8,  add_queue9,  add_queue10,  add_queue11,  add_queue12,  add_queue13,  add_queue14,  add_queue15,  add_queue16;
   
   // this is essentially a 17 x 4 bit array       
   reg [0:67] 	 add_queue; // synthesis ARRAY_UPDATE="RW"
   
   //   reg [0:3] 	 shift_queue0, shift_queue1, shift_queue2, shift_queue3, shift_queue4, shift_queue5, shift_queue6,  shift_queue7,  shift_queue8,  shift_queue9,  shift_queue10,  shift_queue11,  shift_queue12,  shift_queue13,  shift_queue14,  shift_queue15,  shift_queue16;
   
   // this is a 17 x 4 bit array
   reg [0:67] 	 shift_queue;  // synthesis ARRAY_UPDATE="RW" 
   
   //reg [0:61] 	  cmd_block_table0, cmd_block_table1, cmd_block_table2, cmd_block_table3,  cmd_block_table4, cmd_block_table5, cmd_block_table6, cmd_block_table7, cmd_block_table8, cmd_block_table9, cmd_block_table10, cmd_block_table11, cmd_block_table12, cmd_block_table13, cmd_block_table14, cmd_block_table15;

   // these are 8 x 62 bit arrays
   reg [0:495] 	 cmd_block_table1, // synthesis ARRAY_UPDATE="RW"
                 cmd_block_table2; // synthesis ARRAY_UPDATE="RW" 

   reg 		 add_or_shift;
   reg [0:4] branch1, branch2, branch3, branch4;
   reg 	     branch1_cmd,  branch2_cmd, branch3_cmd, branch4_cmd;
   wire [0:3] add_dispatch_pointer, shift_dispatch_pointer;
   wire       rw_case, rw_case0, rw_case1, rw_case2, rw_case3, rw_case4, rw_case5, rw_case6, rw_case7, rw_case8, rw_case9, rw_case10, rw_case11, rw_case12,rw_case13, rw_case14, rw_case15;
   

   always
     @ (negedge c_clk) begin

	/*Set the values of the addqueue for the next cycles
	 The next cycle's value either comes from a new command or from the next position in
	 addqueue.

	 set add_queue(0) to zeros, we will try to access this on cycles with no add is dispatched
	 */

	add_queue[0:3] <= (reset) ? 4'b0 : add_queue[0:3];
	
	add_queue[4:7] <= (reset) ? 4'b0 :
			  (hold1_add_pos[0:4] == 5'b00001) ? {2'b00 , hold1_tag[0:1]}:
			  (hold2_add_pos[0:4] == 5'b00001) ? {2'b01 , hold2_tag[0:1]}:
			  (hold3_add_pos[0:4] == 5'b00001) ? {2'b10, hold3_tag[0:1]}:
			  (hold4_add_pos[0:4] == 5'b00001) ? {2'b11, hold4_tag[0:1]}:
			  ((dispatch_add <= 5'b00001) && (dispatch_add != 5'b0)) ? 
			  add_queue[2*4+0:2*4+3]:  add_queue[1*4+0:1*4+3];
	
	add_queue[2*4+0:2*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b00010) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b00010) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b00010) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b00010) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b00010) && (dispatch_add != 5'b0)) ? 
				  add_queue[3*4+0:3*4+3]:  add_queue[2*4+0:2*4+3];

	add_queue[3*4+0:3*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b00011) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b00011) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b00011) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b00011) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b00011) && (dispatch_add != 5'b0)) ? 
				  add_queue[4*4+0:4*4+3]:  add_queue[3*4+0:3*4+3];

	add_queue[4*4+0:4*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b00100) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b00100) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b00100) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b00100) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b00100) && (dispatch_add != 5'b0)) ? 
				  add_queue[5*4+0:5*4+3]:  add_queue[4*4+0:4*4+3];

	add_queue[5*4+0:5*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b00101) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b00101) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b00101) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b00101) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b00101) && (dispatch_add != 5'b0)) ? 
				  add_queue[6*4+0:6*4+3]:  add_queue[5*4+0:5*4+3];

	add_queue[6*4+0:6*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b00110) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b00110) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b00110) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b00110) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b00110) && (dispatch_add != 5'b0)) ? 
				  add_queue[7*4+0:7*4+3]:  add_queue[6*4+0:6*4+3];

	add_queue[7*4+0:7*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b00111) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b00111) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b00111) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b00111) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b00111) && (dispatch_add != 5'b0)) ? 
				  add_queue[8*4+0:8*4+3]:  add_queue[7*4+0:7*4+3];

	add_queue[8*4+0:8*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b01000) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b01000) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b01000) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b01000) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b01000) && (dispatch_add != 5'b0)) ? 
				  add_queue[9*4+0:9*4+3]:  add_queue[8*4+0:8*4+3];
	
	add_queue[9*4+0:9*4+3] <= (reset) ? 4'b0 :
				  (hold1_add_pos[0:4] == 5'b01001) ? {2'b00 , hold1_tag[0:1]}:
				  (hold2_add_pos[0:4] == 5'b01001) ? {2'b01 , hold2_tag[0:1]}:
				  (hold3_add_pos[0:4] == 5'b01001) ? {2'b10, hold3_tag[0:1]}:
				  (hold4_add_pos[0:4] == 5'b01001) ? {2'b11, hold4_tag[0:1]}:
				  ((dispatch_add <= 5'b01001) && (dispatch_add != 5'b0)) ? 
				  add_queue[10*4+0:10*4+3]:  add_queue[9*4+0:9*4+3];
	
	add_queue[10*4+0:10*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b01010) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b01010) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b01010) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b01010) ? {2'b11, hold4_tag[0:1]}:
				    (((dispatch_add <= 5'b01010) && (dispatch_add != 5'b0)) && skip_11) ? add_queue[12*4+0:12*4+3] :
				    (((dispatch_add <= 5'b01010) && (dispatch_add != 5'b0)) && ~skip_11) ? add_queue[11*4+0:11*4+3] :
				    add_queue[10*4+0:10*4+3];
		
	add_queue[11*4+0:11*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b01011) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b01011) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b01011) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b01011) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_add <= 5'b01011) && (dispatch_add != 5'b0)) ? 
				    add_queue[12*4+0:12*4+3]:  add_queue[11*4+0:12*4+3];
	
	add_queue[12*4+0:12*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b01100) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b01100) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b01100) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b01100) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_add <= 5'b01100) && (dispatch_add != 5'b0)) ? 
				    add_queue[13*4+0:13*4+3]:  add_queue[12*4+0:12*4+3];
	
	add_queue[13*4+0:13*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b01101) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b01101) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b01101) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b01101) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_add <= 5'b01101) && (dispatch_add != 5'b0)) ? 
				    add_queue[14*4+0:14*4+3]:  add_queue[13*4+0:13*4+3];
	
	add_queue[14*4+0:14*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b01110) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b01110) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b01110) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b01110) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_add <= 5'b01110) && (dispatch_add != 5'b0)) ? 
				    add_queue[15*4+0:15*4+3]:  add_queue[14*4+0:14*4+3];
	
	add_queue[15*4+0:15*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b01111) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b01111) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b01111) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b01111) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_add <= 5'b01111) && (dispatch_add != 5'b0)) ? 
				    add_queue[16*4+0:16*4+3]:  add_queue[15*4+0:15*4+3];
	
	add_queue[16*4+0:16*4+3] <= (reset) ? 4'b0 :
				    (hold1_add_pos[0:4] == 5'b10000) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_add_pos[0:4] == 5'b10000) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_add_pos[0:4] == 5'b10000) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_add_pos[0:4] == 5'b10000) ? {2'b11, hold4_tag[0:1]}:
				    4'b00;
     end // always @ (negedge c_clk)

   always
     @ (negedge c_clk) begin

	/*Set the values of the shiftqueue for the next cycles
	 The next cycle's value either comes from a new command or from the next position in
	 shiftqueue.

	 set shift_queue(0) to zeros, we will try to access this on cycles with no shift is dispatched
	 */

	shift_queue[0:3] <= (reset) ? 4'b0 : shift_queue[0:3];
	
	shift_queue[1*4+0:1*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00001) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00001) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00001) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00001) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00001) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[2*4+0:2*4+3]:  shift_queue[1*4+0:1*4+3];
	
	shift_queue[2*4+0:2*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00010) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00010) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00010) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00010) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00010) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[3*4+0:3*4+3]:  shift_queue[2*4+0:2*4+3];
	
	shift_queue[3*4+0:3*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00011) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00011) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00011) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00011) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00011) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[4*4+0:4*4+3]:  shift_queue[3*4+0:3*4+3];
	
	shift_queue[4*4+0:4*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00100) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00100) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00100) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00100) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00100) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[5*4+0:5*4+3]:  shift_queue[4*4+0:4*4+3];
	
	shift_queue[5*4+0:5*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00101) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00101) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00101) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00101) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00101) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[6*4+0:6*4+3]:  shift_queue[5*4+0:5*4+3];
	
	shift_queue[6*4+0:6*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00110) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00110) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00110) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00110) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00110) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[7*4+0:7*4+3]:  shift_queue[6*4+0:6*4+3];
	
	shift_queue[7*4+0:7*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b00111) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b00111) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b00111) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b00111) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b00111) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[8*4+0:8*4+3]:  shift_queue[7*4+0:7*4+3];
	
	shift_queue[8*4+0:8*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b01000) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b01000) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b01000) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b01000) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b01000) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[9*4+0:9*4+3]:  shift_queue[8*4+0:8*4+3];
	
	shift_queue[9*4+0:9*4+3] <= (reset) ? 4'b0 :
				    (hold1_shift_pos[0:4] == 5'b01001) ? {2'b00 , hold1_tag[0:1]}:
				    (hold2_shift_pos[0:4] == 5'b01001) ? {2'b01 , hold2_tag[0:1]}:
				    (hold3_shift_pos[0:4] == 5'b01001) ? {2'b10, hold3_tag[0:1]}:
				    (hold4_shift_pos[0:4] == 5'b01001) ? {2'b11, hold4_tag[0:1]}:
				    ((dispatch_shift <= 5'b01001) && (dispatch_shift != 5'b0)) ? 
				    shift_queue[10*4+0:10*4+3]:  shift_queue[9*4+0:9*4+3];
	
	shift_queue[10*4+0:10*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b01010) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b01010) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b01010) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b01010) ? {2'b11, hold4_tag[0:1]}:
				      ((dispatch_shift <= 5'b01010) && (dispatch_shift != 5'b0)) ? 
				      shift_queue[11*4+0:11*4+3]:  shift_queue[10*4+0:10*4+3];
	
	shift_queue[11*4+0:11*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b01011) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b01011) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b01011) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b01011) ? {2'b11, hold4_tag[0:1]}:
				      ((dispatch_shift <= 5'b01011) && (dispatch_shift != 5'b0)) ? 
				      shift_queue[12*4+0:12*4+3]:  shift_queue[11*4+0:12*4+3];
	
	shift_queue[12*4+0:12*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b01100) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b01100) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b01100) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b01100) ? {2'b11, hold4_tag[0:1]}:
				      ((dispatch_shift <= 5'b01100) && (dispatch_shift != 5'b0)) ? 
				      shift_queue[13*4+0:13*4+3]:  shift_queue[12*4+0:12*4+3];
	
	shift_queue[13*4+0:13*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b01101) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b01101) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b01101) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b01101) ? {2'b11, hold4_tag[0:1]}:
				      ((dispatch_shift <= 5'b01101) && (dispatch_shift != 5'b0)) ? 
				      shift_queue[14*4+0:14*4+3]:  shift_queue[13*4+0:13*4+3];
	
	shift_queue[14*4+0:14*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b01110) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b01110) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b01110) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b01110) ? {2'b11, hold4_tag[0:1]}:
				      ((dispatch_shift <= 5'b01110) && (dispatch_shift != 5'b0)) ? 
				      shift_queue[15*4+0:15*4+3]:  shift_queue[14*4+0:14*4+3];
	
	shift_queue[15*4+0:15*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b01111) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b01111) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b01111) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b01111) ? {2'b11, hold4_tag[0:1]}:
				      ((dispatch_shift <= 5'b01111) && (dispatch_shift != 5'b0)) ? 
				      shift_queue[16*4+0:16*4+3]:  shift_queue[15*4+0:15*4+3];
	
	shift_queue[16*4+0:16*4+3] <= (reset) ? 4'b0 :
				      (hold1_shift_pos[0:4] == 5'b10000) ? {2'b00 , hold1_tag[0:1]}:
				      (hold2_shift_pos[0:4] == 5'b10000) ? {2'b01 , hold2_tag[0:1]}:
				      (hold3_shift_pos[0:4] == 5'b10000) ? {2'b10, hold3_tag[0:1]}:
				      (hold4_shift_pos[0:4] == 5'b10000) ? {2'b11, hold4_tag[0:1]}:
				      4'b00;
     end // always @ (negedge c_clk)

   always
     @ (negedge c_clk) begin
	//branch1-4 are latched when a branch cmd comes in, so the
	// next cmd coming in on the same port knows whether or not
	// it follows a branch, and if it does, it knows the branch's tag

	branch1 <= (reset) ? 5'b0 : 
		   ((hold1_cmd == 'b1100) || (hold1_cmd == 'b1101)) ? {3'b100 ,hold1_tag[0:1]} :
		   ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1)) ? 5'b0 : branch1;
	
	branch2 <= (reset) ? 5'b0 : 
		   ((hold2_cmd == 'b1100) || (hold2_cmd == 'b1101)) ? {3'b101 , hold2_tag[0:1]} :
		   ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1)) ? 5'b0 : branch2;

	branch3 <= (reset) ? 5'b0 : 
		   ((hold3_cmd == 'b1100) || (hold3_cmd == 'b1101)) ? {3'b110 , hold3_tag[0:1]} :
		   ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1)) ? 5'b0 : branch3;

	branch4 <= (reset) ? 5'b0 : 
		   ((hold4_cmd == 'b1100) || (hold4_cmd == 'b1101)) ? {3'b111 , hold4_tag[0:1]} :
		   ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1)) ? 5'b0 : branch4;

	//branch1-4_cmd are latched when a cmd following a branch
	// comes in, so the cmd knows it follows a branch
	 
	branch1_cmd <= (reset) ? 1'b0 :
		       ( branch1[0] && (hold1_cmd == 4'b1100) && (hold1_cmd == 4'b1101)) ? 1'b1 : 1'b0;

	branch2_cmd <= (reset) ? 1'b0 :
		       ( branch2[0] && (hold2_cmd == 4'b1100) && (hold2_cmd == 4'b1101)) ? 1'b1 : 1'b0;

	branch3_cmd <= (reset) ? 1'b0 :
		       ( branch3[0] && (hold3_cmd == 4'b1100) && (hold3_cmd == 4'b1101)) ? 1'b1 : 1'b0;
	
	branch4_cmd <= (reset) ? 1'b0 :
		       ( branch4[0] && (hold4_cmd == 4'b1100) && (hold4_cmd == 4'b1101)) ? 1'b1 : 1'b0;

	/**********************
	
	 the cmd_block_table uses the following bits to store information about each
	  cmd, plus information on whether or not it is
	 valid, which other cmds are blocking it from being dispatched, and the
	  tag of the branch it follows (if it follows a branch)
	 
	 --bit 0         - valid bit
	 --bits 1 to 3   - blocking bits set if other cmds on same port must go before this cmd
	 --bit 4         - blocking bit if previous cmd dispatched writes to same register that this cmd reads from
	 --bits 5 to 9   - follows branch tag, bit 5 = valid bit
	 --bits 10 to 13 - cmd
	 --bits 14 to 18 - data1 register, bit 14 = valid bit
	 --bits 19 to 23 - data2 register, bit 19 = valid bit
	 --bits 24 to 28 - result register, bit 24 = valid bit
	 --bits 29 to 60 - input data (for stores)
	 --bit 61        - set to 0 for add/subtract/branch cmds, set to 1 for shift/store/fetch cmds
*/


//	 ----------------------------------
//	 -- cmd_block_table for requestor 1
//	 ----------------------------------

//	 -- tag 00
	 
	cmd_block_table1[0*62+0] <= (reset) ? 1'b0 : 
				    ( (hold1_tag == 2'b00) && ( (hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) && (dispatch_add != 5'b0)) || 
				     (( ~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ) && (dispatch_shift != 5'b0))) ? 
				    1'b0:
				    cmd_block_table1[0*62+0];

	cmd_block_table1[0*62+1] <= (reset) ? 1'b0 :
				   ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[1*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_result == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_result == cmd_block_table1[1*62+15:1*62+18]) && cmd_block_table1[1*62+14]) || 
				     ((hold1_result == cmd_block_table1[1*62+20:1*62+23]) && cmd_block_table1[1*62+19]) || 
				     (branch1 == 5'b10001) || 
				     (((hold1_cmd == 'b1100) || (hold1_cmd == 'b1101)) && (cmd_block_table1[1*62+5:1*62+9] == 5'b10000) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[0*62+1];

	cmd_block_table1[0*62+2] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[2*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_result == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_result == cmd_block_table1[2*62+15:2*62+18]) && cmd_block_table1[2*62+14]) || 
				     ((hold1_result == cmd_block_table1[2*62+20:2*62+23]) && cmd_block_table1[2*62+19]) || 
				     (branch1 == 5'b10010) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[2*62+5:2*62+9] == 5'b10000) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				   cmd_block_table1[0*62+2];

	cmd_block_table1[0*62+3] <= (reset) ? 1'b0 :
				   ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[3*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_result == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_result == cmd_block_table1[3*62+15:3*62+18]) && cmd_block_table1[3*62+14]) || 
				     ((hold1_result == cmd_block_table1[3*62+20:3*62+23]) && cmd_block_table1[3*62+19]) || 
				     (branch1 == 5'b10011) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[3*62+5:3*62+9] == 5'b10000) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[0*62+3];

	cmd_block_table1[0*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table1[0*62+14:0*62+18]) && cmd_block_table1[0*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table1[0*62+19:0*62+23]) && cmd_block_table1[0*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[0*62+14:0*62+18]) && cmd_block_table1[0*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[0*62+19:0*62+23]) && cmd_block_table1[0*62+19]) ||
				     rw_case0) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table1[0*62+14:0*62+18]) || ~cmd_block_table1[0*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table1[0*62+19:0*62+23]) || ~cmd_block_table1[0*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[0*62+14:0*62+18]) || ~cmd_block_table1[0*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[0*62+19:0*62+23]) || ~cmd_block_table1[0*62+19]) &&
				     ~rw_case0) ? 1'b0 :
				    cmd_block_table1[0*62+4];
	
	cmd_block_table1[0*62+5:0*62+9] <= (reset) ? 5'b0 :
					   ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? branch1 :
					   cmd_block_table1[0*62+5:0*62+9];
	
	cmd_block_table1[0*62+10:0*62+13] <= (reset) ? 4'b0 :
					     ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? hold1_cmd :
					     cmd_block_table1[0*62+10:0*62+13];

	cmd_block_table1[0*62+14:0*62+18] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b0) && (hold1_cmd != 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data1} :
					     ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[0*62+14:0*62+18];

	cmd_block_table1[0*62+19:0*62+23] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b0) && (hold1_cmd != 'b1001) && (hold1_cmd!= 'b1010) && (hold1_cmd != 'b1100) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data2} :
					     ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[0*62+19:0*62+23];

	cmd_block_table1[0*62+24:0*62+28] <= (reset) ? 5'b0 :
					    ((hold1_tag == 2'b0) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) && (hold1_cmd != 'b1101) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_result} :
					    ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[0*62+24:0*62+28];

	cmd_block_table1[0*62+29:0*62+60] <= (reset) ? 32'b0 :
					    ((hold1_tag == 2'b0) && (hold1_cmd == 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? hold1_data :
					    ((hold1_tag == 2'b0) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[0*62+29:0*62+60];

	cmd_block_table1[0*62+61] <= (reset) ? 'b0 :
				     ((hold1_tag == 2'b0) && (hold1_add_valid == 5'b1)) ? 'b0 :
				     ((hold1_tag == 2'b0) && (hold1_shift_valid == 5'b1)) ? 'b1 :
				     cmd_block_table1[0*62+61];

	// tag 01

	cmd_block_table1[1*62+0] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b1) && ( (hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     (( ~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table1[1*62+0];

	cmd_block_table1[1*62+1] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[0*62+0] && 
				    (!(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]))  && 
				    (!(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold1_data1 == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_result == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_result == cmd_block_table1[0*62+15:0*62+18]) && cmd_block_table1[0*62+14]) || 
				     ((hold1_result == cmd_block_table1[0*62+20:0*62+23]) && cmd_block_table1[0*62+19]) || 
				     (branch1 == 5'b10000) || 
				     (((hold1_cmd == 'b1100) || (hold1_cmd == 'b1101)) && (cmd_block_table1[0*62+5:0*62+9] == 5'b10001) && branch1_cmd)) :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) && (dispatch_add != 5'b0)) || 
				     ((~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && (dispatch_shift != 5'b0))) ? 
				    1'b0 :
				    cmd_block_table1[1*62+1];

	cmd_block_table1[1*62+2] <= (reset) ? 1'b0 :
				   ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[2*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_result == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_result == cmd_block_table1[2*62+15:2*62+18]) && cmd_block_table1[2*62+14]) || 
				     ((hold1_result == cmd_block_table1[2*62+20:2*62+23]) && cmd_block_table1[2*62+19]) || 
				     (branch1 == 5'b10010) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[2*62+5:2*62+9] == 5'b10001) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[1*62+2];

	cmd_block_table1[1*62+3] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[3*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_result == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_result == cmd_block_table1[3*62+15:3*62+18]) && cmd_block_table1[3*62+14]) || 
				     ((hold1_result == cmd_block_table1[3*62+20:3*62+23]) && cmd_block_table1[3*62+19]) || 
				     (branch1 == 5'b10011) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[3*62+5:3*62+9] == 5'b10001) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[1*62+3];

	cmd_block_table1[1*62+4] <= (reset) ? 1'b0 :
				  (((rd_wr_conflict_block1 == cmd_block_table1[1*62+14:1*62+18]) && cmd_block_table1[1*62+14]) ||
				   ((rd_wr_conflict_block1 == cmd_block_table1[1*62+19:1*62+23]) && cmd_block_table1[1*62+19]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table1[1*62+14:1*62+18]) && cmd_block_table1[1*62+14]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table1[1*62+19:1*62+23]) && cmd_block_table1[1*62+19]) ||
				   rw_case1) ?  1'b1 :
				  (((rd_wr_conflict_block1 != cmd_block_table1[1*62+14:1*62+18]) || ~cmd_block_table1[1*62+14]) &&
				   ((rd_wr_conflict_block1 != cmd_block_table1[1*62+19:1*62+23]) || ~cmd_block_table1[1*62+19]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table1[1*62+14:1*62+18]) || ~cmd_block_table1[1*62+14]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table1[1*62+19:1*62+23]) || ~cmd_block_table1[1*62+19]) &&
				   ~rw_case1) ? 1'b0 :
				  cmd_block_table1[1*62+4];

	cmd_block_table1[1*62+5:1*62+9] <= (reset) ? 5'b0 :
					   ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? branch1 :
					   cmd_block_table1[1*62+5:1*62+9];

	cmd_block_table1[1*62+10:1*62+13] <= (reset) ? 4'b0 :
					    ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? hold1_cmd :
					    cmd_block_table1[1*62+10:1*62+13];

	cmd_block_table1[1*62+14:1*62+18] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b1) && (hold1_cmd != 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data1} :
					     ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[1*62+14:1*62+18];

	cmd_block_table1[1*62+19:1*62+23] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b1) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data2} :
					     ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[1*62+19:1*62+23];

	cmd_block_table1[1*62+24:1*62+28] <= (reset) ? 5'b0 :
					    ((hold1_tag == 2'b1) && (hold1_cmd != 'b1010) && (hold1_cmd!= 'b1100) && (hold1_cmd != 'b1101) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_result} :
					    ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[1*62+24:1*62+28];

	cmd_block_table1[1*62+29:1*62+60] <= (reset) ? 32'b0 :
					    ((hold1_tag == 2'b1) && (hold1_cmd == 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? hold1_data :
					    ((hold1_tag == 2'b1) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[1*62+29:1*62+60];

	cmd_block_table1[1*62+61] <= (reset) ? 1'b0 :
				  ((hold1_tag == 2'b1) && (hold1_add_valid == 5'b1)) ? 1'b0 :
				  ((hold1_tag == 2'b1) && (hold1_shift_valid == 5'b1)) ? 1'b1 :
				  cmd_block_table1[1*62 + 61];
	
	// tag 10
	
	cmd_block_table1[2*62+0] <= (reset) ? 1'b0 :
				    ( (hold1_tag == 2'b10) && ( (hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 1'b1 :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) )  ||  
				     (( ~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ) )) ? 
				    1'b0:
				    cmd_block_table1[2*62+0];

	cmd_block_table1[2*62+1] <= (reset) ? 1'b0 :
				   ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[0*62+0] && 
				    (!(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || (dispatch_add == 5'b0)) && 
				    (!(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) || (dispatch_shift == 5'b0)) && 
				    (((hold1_data1 == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_result == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_result == cmd_block_table1[0*62+15:0*62+18]) && cmd_block_table1[0*62+14]) || 
				     ((hold1_result == cmd_block_table1[0*62+20:0*62+23]) && cmd_block_table1[0*62+19]) || 
				     (branch1 == 5'b10000) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[0*62+5:0*62+9] == 5'b10010) && branch1_cmd)) :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) && (dispatch_add != 5'b0)) || 
				     ((~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && (dispatch_shift != 5'b0))) ? 
				    1'b0 :
				    cmd_block_table1[2*62+1];

	cmd_block_table1[2*62+2] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[1*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_result == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_result == cmd_block_table1[1*62+15:1*62+18]) && cmd_block_table1[1*62+14]) || 
				     ((hold1_result == cmd_block_table1[1*62+20:1*62+23]) && cmd_block_table1[1*62+19]) || 
				     (branch1 == 5'b10001) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[1*62+5:1*62+9] == 5'b10010) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[2*62+2];

	cmd_block_table1[2*62+3] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[3*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_result == cmd_block_table1[3*62+25:3*62+28]) && cmd_block_table1[3*62+24]) || 
				     ((hold1_result == cmd_block_table1[3*62+15:3*62+18]) && cmd_block_table1[3*62+14]) || 
				     ((hold1_result == cmd_block_table1[3*62+20:3*62+23]) && cmd_block_table1[3*62+19]) || 
				     (branch1 == 5'b10011) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[3*62+5:3*62+9] == 5'b10010) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[2*62+3];

	cmd_block_table1[2*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table1[2*62+14:2*62+18]) && cmd_block_table1[2*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table1[2*62+19:2*62+23]) && cmd_block_table1[2*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[2*62+14:2*62+18]) && cmd_block_table1[2*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[2*62+19:2*62+23]) && cmd_block_table1[2*62+19]) ||
				     rw_case2) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table1[2*62+14:2*62+18]) || ~cmd_block_table1[2*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table1[2*62+19:2*62+23]) || ~cmd_block_table1[2*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[2*62+14:2*62+18]) || ~cmd_block_table1[2*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[2*62+19:2*62+23]) || ~cmd_block_table1[2*62+19]) &&
				     ~rw_case2) ? 1'b0 :
				    cmd_block_table1[2*62+4];

	cmd_block_table1[2*62+5:2*62+9] <= (reset) ? 5'b0 :
					   ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? branch1 :
					   cmd_block_table1[2*62+5:2*62+9];
	
	cmd_block_table1[2*62+10:2*62+13] <= (reset) ? 4'b0 :
					     ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? hold1_cmd :
					     cmd_block_table1[2*62+10:2*62+13];
	
	cmd_block_table1[2*62+14:2*62+18] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b10) && (hold1_cmd != 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data1} :
					     ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[2*62+14:2*62+18];
	
	cmd_block_table1[2*62+19:2*62+23] <= (reset) ? 5'b0 :
					   ((hold1_tag == 2'b10) && (hold1_cmd != 'b1001) && (hold1_cmd!= 'b1010) && (hold1_cmd != 'b1100) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data2} :
					    ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table1[2*62+19:2*62+23];

	cmd_block_table1[2*62+24:2*62+28] <= (reset) ? 5'b0 :
					    ((hold1_tag == 2'b10) && (hold1_cmd != 'b1010) && (hold1_cmd!= 'b1100) && (hold1_cmd != 'b1101) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_result} :
					    ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[2*62+24:2*62+28];

	cmd_block_table1[2*62+29:2*62+60] <= (reset) ? 32'b0 :
					    ((hold1_tag == 2'b10) && (hold1_cmd == 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? hold1_data :
					    ((hold1_tag == 2'b10) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[2*62+29:2*62+60];

	cmd_block_table1[2*62+61] <= (reset) ? 'b0 :
				  ((hold1_tag == 2'b10) && (hold1_add_valid == 5'b1)) ? 'b0 :
				  ((hold1_tag == 2'b10) && (hold1_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table1[2*62 + 61];


	// tag 11

	cmd_block_table1[3*62+0] <= (reset) ? 1'b0 :
				    ( (hold1_tag == 2'b11) && ( (hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 1'b1 :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     (( ~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table1[3*62+0];

	cmd_block_table1[3*62+1] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[0*62+0] && 
				    (!(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || (dispatch_add == 5'b0))  && 
				    (!(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) || (dispatch_shift == 5'b0)) && 
				    (((hold1_data1 == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_result == cmd_block_table1[0*62+25:0*62+28]) && cmd_block_table1[0*62+24]) || 
				     ((hold1_result == cmd_block_table1[0*62+15:0*62+18]) && cmd_block_table1[0*62+14]) || 
				     ((hold1_result == cmd_block_table1[0*62+20:0*62+23]) && cmd_block_table1[0*62+19]) || 
				     (branch1 == 5'b10000) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[0*62+5:0*62+9] == 5'b10011) && branch1_cmd)) :
				    (((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) && (dispatch_add != 5'b0)) ||
				     ((~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && (dispatch_shift != 5'b0))) ? 
				    1'b0 :
				    cmd_block_table1[3*62+1];

	cmd_block_table1[3*62+2] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[1*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_result == cmd_block_table1[1*62+25:1*62+28]) && cmd_block_table1[1*62+24]) || 
				     ((hold1_result == cmd_block_table1[1*62+15:1*62+18]) && cmd_block_table1[1*62+14]) || 
				     ((hold1_result == cmd_block_table1[1*62+20:1*62+23]) && cmd_block_table1[1*62+19]) || 
				     (branch1 == 5'b10001) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[1*62+5:1*62+9] == 5'b10011) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[3*62+2];

	cmd_block_table1[3*62+3] <= (reset) ? 1'b0 :
				    ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ? cmd_block_table1[2*62+0] && 
				    !(~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold1_data1 == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_data2 == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_result == cmd_block_table1[2*62+25:2*62+28]) && cmd_block_table1[2*62+24]) || 
				     ((hold1_result == cmd_block_table1[2*62+15:2*62+18]) && cmd_block_table1[2*62+14]) || 
				     ((hold1_result == cmd_block_table1[2*62+20:2*62+23]) && cmd_block_table1[2*62+19]) || 
				     (branch1 == 5'b10010) || 
				     (((hold1_cmd == 'b1100) || hold1_cmd == 'b1101) && (cmd_block_table1[2*62+5:2*62+9] == 5'b10011) && branch1_cmd)) :
				    ((~add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[3*62+3];

	cmd_block_table1[3*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table1[3*62+14:3*62+18]) && cmd_block_table1[3*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table1[3*62+19:3*62+23]) && cmd_block_table1[3*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[3*62+14:3*62+18]) && cmd_block_table1[3*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[3*62+19:3*62+23]) && cmd_block_table1[3*62+19]) ||
				     rw_case3) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table1[3*62+14:3*62+18]) || ~cmd_block_table1[3*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table1[3*62+19:3*62+23]) || ~cmd_block_table1[3*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[3*62+14:3*62+18]) || ~cmd_block_table1[3*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[3*62+19:3*62+23]) || ~cmd_block_table1[3*62+19]) &&
				   ~rw_case3) ? 1'b0 :
				    cmd_block_table1[3*62+4];
	
	cmd_block_table1[3*62+5:3*62+9] <= (reset) ? 5'b0 :
					  ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? branch1 :
					  cmd_block_table1[3*62+5:3*62+9];

	cmd_block_table1[3*62+10:3*62+13] <= (reset) ? 4'b0 :
					     ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? hold1_cmd :
					     cmd_block_table1[3*62+10:3*62+13];

	cmd_block_table1[3*62+14:3*62+18] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b11) && (hold1_cmd != 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data1} :
					     ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[3*62+14:3*62+18];

	cmd_block_table1[3*62+19:3*62+23] <= (reset) ? 5'b0 :
					     ((hold1_tag == 2'b11) && (hold1_cmd != 'b1001) && (hold1_cmd!= 'b1010) && (hold1_cmd != 'b1100) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_data2} :
					     ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[3*62+19:3*62+23];

	cmd_block_table1[3*62+24:3*62+28] <= (reset) ? 5'b0 :
					    ((hold1_tag == 2'b11) && (hold1_cmd != 'b1010) && (hold1_cmd!= 'b1100) && (hold1_cmd != 'b1101) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? {1'b1, hold1_result} :
					    ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b00001) || (hold1_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[3*62+24:3*62+28];

	cmd_block_table1[3*62+29:3*62+60] <= (reset) ? 32'b0 :
					    ((hold1_tag == 2'b11) && (hold1_cmd == 'b1001) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? hold1_data :
					    ((hold1_tag == 2'b11) && ((hold1_add_valid == 5'b1 || hold1_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[3*62+29:3*62+60];

	cmd_block_table1[3*62+61] <= (reset) ? 'b0 :
				  ((hold1_tag == 2'b11) && (hold1_add_valid == 5'b1)) ? 'b0 :
				  ((hold1_tag == 2'b11) && (hold1_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table1[3*62 + 61];


//	 ----------------------------------
//	 -- cmd_block_table for requestor 2
//	 ----------------------------------

//	 -- tag 00
	 
	cmd_block_table1[4*62+0] <= (reset) ? 1'b0 : 
				    ((hold2_tag == 2'b00) && ( (hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     (( ~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table1[4*62+0];

	cmd_block_table1[4*62+1] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[5*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_result == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_result == cmd_block_table1[5*62+15:5*62+18]) && cmd_block_table1[5*62+14]) || 
				     ((hold2_result == cmd_block_table1[5*62+20:5*62+23]) && cmd_block_table1[5*62+19]) || 
				     (branch2 == 5'b10101) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[5*62+5:5*62+9] == 5'b10100) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[4*62+1];

	cmd_block_table1[4*62+2] <= (reset) ? 1'b0 :
				   ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[6*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_result == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_result == cmd_block_table1[6*62+15:6*62+18]) && cmd_block_table1[6*62+14]) || 
				     ((hold2_result == cmd_block_table1[6*62+20:6*62+23]) && cmd_block_table1[6*62+19]) || 
				     (branch2 == 5'b10110) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[6*62+5:6*62+9] == 5'b10100) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				   cmd_block_table1[4*62+2];

	cmd_block_table1[4*62+3] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[7*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_result == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_result == cmd_block_table1[7*62+15:7*62+18]) && cmd_block_table1[7*62+14]) || 
				     ((hold2_result == cmd_block_table1[7*62+20:7*62+23]) && cmd_block_table1[7*62+19]) || 
				     (branch2 == 5'b10111) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[7*62+5:7*62+9] == 5'b10100) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[4*62+3];

	cmd_block_table1[4*62+4] <= (reset) ? 1'b0 :
				  (((rd_wr_conflict_block1 == cmd_block_table1[4*62+14:4*62+18]) && cmd_block_table1[4*62+14]) ||
				   ((rd_wr_conflict_block1 == cmd_block_table1[4*62+19:4*62+23]) && cmd_block_table1[4*62+19]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table1[4*62+14:4*62+18]) && cmd_block_table1[4*62+14]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table1[4*62+19:4*62+23]) && cmd_block_table1[4*62+19]) ||
				   rw_case4) ?  1'b1 :
				  (((rd_wr_conflict_block1 != cmd_block_table1[4*62+14:4*62+18]) || ~cmd_block_table1[4*62+14]) &&
				   ((rd_wr_conflict_block1 != cmd_block_table1[4*62+19:4*62+23]) || ~cmd_block_table1[4*62+19]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table1[4*62+14:4*62+18]) || ~cmd_block_table1[4*62+14]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table1[4*62+19:4*62+23]) || ~cmd_block_table1[4*62+19]) &&
				   ~rw_case4) ? 1'b0 :
				  cmd_block_table1[4*62+4];

	cmd_block_table1[4*62+5:4*62+9] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? branch2 :
					   cmd_block_table1[4*62+5:4*62+9];

	cmd_block_table1[4*62+10:4*62+13] <= (reset) ? 4'b0 :
					    ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? hold2_cmd :
					    cmd_block_table1[4*62+10:4*62+13];

	cmd_block_table1[4*62+14:4*62+18] <= (reset) ? 5'b0 :
					     ((hold2_tag == 2'b0) && (hold2_cmd != 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data1} :
					     ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[4*62+14:4*62+18];

	cmd_block_table1[4*62+19:4*62+23] <= (reset) ? 5'b0 :
					     ((hold2_tag == 2'b0) && (hold2_cmd != 'b1001) && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data2} :
					     ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[4*62+19:4*62+23];

	cmd_block_table1[4*62+24:4*62+28] <= (reset) ? 5'b0 :
					    ((hold2_tag == 2'b0) && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && (hold2_cmd != 'b1101) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_result} :
					    ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[4*62+24:4*62+28];

	cmd_block_table1[4*62+29:4*62+60] <= (reset) ? 32'b0 :
					    ((hold2_tag == 2'b0) && (hold2_cmd == 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? hold2_data :
					    ((hold2_tag == 2'b0) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[4*62+29:4*62+60];

	cmd_block_table1[4*62+61] <= (reset) ? 'b0 :
				  ((hold2_tag == 2'b0) && (hold2_add_valid == 5'b1)) ? 'b0 :
				  ((hold2_tag == 2'b0) && (hold2_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table1[4*62+61];

	// tag 01

	cmd_block_table1[5*62+0] <= (reset) ? 1'b0 :
				    ( (hold2_tag == 2'b1) && ( (hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     (( ~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table1[5*62+0];

	cmd_block_table1[5*62+1] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[4*62+0] && 
				    (!(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]))  && 
				    (!(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold2_data1 == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_result == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_result == cmd_block_table1[4*62+15:4*62+18]) && cmd_block_table1[4*62+14]) || 
				     ((hold2_result == cmd_block_table1[4*62+20:4*62+23]) && cmd_block_table1[4*62+19]) || 
				     (branch2 == 5'b10100) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[4*62+5:4*62+9] == 5'b10101) && branch2_cmd)) :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     ((~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table1[5*62+1];

	cmd_block_table1[5*62+2] <= (reset) ? 1'b0 : ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[6*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_result == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_result == cmd_block_table1[6*62+15:6*62+18]) && cmd_block_table1[6*62+14]) || 
				     ((hold2_result == cmd_block_table1[6*62+20:6*62+23]) && cmd_block_table1[6*62+19]) || 
				     (branch2 == 5'b10110) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[6*62+5:6*62+9] == 5'b10101) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[5*62+2];

	cmd_block_table1[5*62+3] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[7*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_result == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_result == cmd_block_table1[7*62+15:7*62+18]) && cmd_block_table1[7*62+14]) || 
				     ((hold2_result == cmd_block_table1[7*62+20:7*62+23]) && cmd_block_table1[7*62+19]) || 
				     (branch2 == 5'b10111) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[7*62+5:7*62+9] == 5'b10101) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[5*62+3];

	cmd_block_table1[5*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table1[5*62+14:5*62+18]) && cmd_block_table1[5*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table1[5*62+19:5*62+23]) && cmd_block_table1[5*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[5*62+14:5*62+18]) && cmd_block_table1[5*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[5*62+19:5*62+23]) && cmd_block_table1[5*62+19]) ||
				     rw_case5) ?  
				    1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table1[5*62+14:5*62+18]) || ~cmd_block_table1[5*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table1[5*62+19:5*62+23]) || ~cmd_block_table1[5*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[5*62+14:5*62+18]) || ~cmd_block_table1[5*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[5*62+19:5*62+23]) || ~cmd_block_table1[5*62+19]) &&
				     ~rw_case5) ? 
				    1'b0 :
				    cmd_block_table1[5*62+4];

	cmd_block_table1[5*62+5:5*62+9] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? branch2 :
					   cmd_block_table1[5*62+5:5*62+9];

	cmd_block_table1[5*62+10:5*62+13] <= (reset) ? 4'b0 :
					    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? hold2_cmd :
					    cmd_block_table1[5*62+10:5*62+13];

	cmd_block_table1[5*62+14:5*62+18] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b1) && (hold2_cmd != 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data1} :
					    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table1[5*62+14:5*62+18];

	cmd_block_table1[5*62+19:5*62+23] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b1) && (hold2_cmd != 'b1001) && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data2} :
					    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table1[5*62+19:5*62+23];

	cmd_block_table1[5*62+24:5*62+28] <= (reset) ? 5'b0 :
					    ((hold2_tag == 2'b1) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) && (hold2_cmd != 'b1101) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_result} :
					    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[5*62+24:5*62+28];

	cmd_block_table1[5*62+29:5*62+60] <= (reset) ? 32'b0 :
					    ((hold2_tag == 2'b1) && (hold2_cmd == 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? hold2_data :
					    ((hold2_tag == 2'b1) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[5*62+29:5*62+60];

	cmd_block_table1[5*62+61] <= (reset) ? 'b0 :
				  ((hold2_tag == 2'b1) && (hold2_add_valid == 5'b1)) ? 'b0 :
				  ((hold2_tag == 2'b1) && (hold2_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table1[5*62 + 61];
	
	// tag 10
	
	cmd_block_table1[6*62+0] <= (reset) ? 1'b0 :
				    ( (hold2_tag == 2'b10) && ( (hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 1'b1 :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) )  ||  
				     (( ~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ) )) ? 
				    1'b0:
				    cmd_block_table1[6*62+0];

	cmd_block_table1[6*62+1] <= (reset) ? 1'b0 :
				   ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[4*62+0] && 
				    (!(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) && 
				    (!(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold2_data1 == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_result == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_result == cmd_block_table1[4*62+15:4*62+18]) && cmd_block_table1[4*62+14]) || 
				     ((hold2_result == cmd_block_table1[4*62+20:4*62+23]) && cmd_block_table1[4*62+19]) || 
				     (branch2 == 5'b10100) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[4*62+5:4*62+9] == 5'b10110) && branch2_cmd)) :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     ((~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])))? 
				    1'b0 :
				    cmd_block_table1[6*62+1];
	
	cmd_block_table1[6*62+2] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[5*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_result == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_result == cmd_block_table1[5*62+15:5*62+18]) && cmd_block_table1[5*62+14]) || 
				     ((hold2_result == cmd_block_table1[5*62+20:5*62+23]) && cmd_block_table1[5*62+19]) || 
				     (branch2 == 5'b10101) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[5*62+5:5*62+9] == 5'b10110) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[6*62+2];
				     
	cmd_block_table1[6*62+3] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[7*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_result == cmd_block_table1[7*62+25:7*62+28]) && cmd_block_table1[7*62+24]) || 
				     ((hold2_result == cmd_block_table1[7*62+15:7*62+18]) && cmd_block_table1[7*62+14]) || 
				     ((hold2_result == cmd_block_table1[7*62+20:7*62+23]) && cmd_block_table1[7*62+19]) || 
				     (branch2 == 5'b10111) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[7*62+5:7*62+9] == 5'b10110) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[6*62+3];

	cmd_block_table1[6*62+4] <= (reset) ? 1'b0 :
				     (((rd_wr_conflict_block1 == cmd_block_table1[6*62+14:6*62+18]) && cmd_block_table1[6*62+14]) ||
				      ((rd_wr_conflict_block1 == cmd_block_table1[6*62+19:6*62+23]) && cmd_block_table1[6*62+19]) ||
				      ((rd_wr_conflict_block2 == cmd_block_table1[6*62+14:6*62+18]) && cmd_block_table1[6*62+14]) ||
				      ((rd_wr_conflict_block2 == cmd_block_table1[6*62+19:6*62+23]) && cmd_block_table1[6*62+19]) ||
				      rw_case6) ?  1'b1 :
				     (((rd_wr_conflict_block1 != cmd_block_table1[6*62+14:6*62+18]) || ~cmd_block_table1[6*62+14]) &&
				      ((rd_wr_conflict_block1 != cmd_block_table1[6*62+19:6*62+23]) || ~cmd_block_table1[6*62+19]) &&
				      ((rd_wr_conflict_block2 != cmd_block_table1[6*62+14:6*62+18]) || ~cmd_block_table1[6*62+14]) &&
				      ((rd_wr_conflict_block2 != cmd_block_table1[6*62+19:6*62+23]) || ~cmd_block_table1[6*62+19]) &&
				      ~rw_case6) ? 1'b0 :
				    cmd_block_table1[6*62+4];
	
	cmd_block_table1[6*62+5:6*62+9] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? branch2 :
					   cmd_block_table1[6*62+5:6*62+9];
	
	cmd_block_table1[6*62+10:6*62+13] <= (reset) ? 4'b0 :
					     ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? hold2_cmd :
					     cmd_block_table1[6*62+10:6*62+13];
	
	cmd_block_table1[6*62+14:6*62+18] <= (reset) ? 5'b0 :
					     ((hold2_tag == 2'b10) && (hold2_cmd != 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data1} :
					     ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[6*62+14:6*62+18];
	
	cmd_block_table1[6*62+19:6*62+23] <= (reset) ? 5'b0 :
					     ((hold2_tag == 2'b10) && (hold2_cmd != 'b1001) && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data2} :
					     ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[6*62+19:6*62+23];
	
	cmd_block_table1[6*62+24:6*62+28] <= (reset) ? 5'b0 :
					     ((hold2_tag == 2'b10)  && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && (hold2_cmd != 'b1101) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_result} :
					     ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table1[6*62+24:6*62+28];
	
	cmd_block_table1[6*62+29:6*62+60] <= (reset) ? 32'b0 :
					     ((hold2_tag == 2'b10) && (hold2_cmd == 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? hold2_data :
					     ((hold2_tag == 2'b10) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? 32'b0 :
					     cmd_block_table1[6*62+29:6*62+60];
	
	cmd_block_table1[6*62+61] <= (reset) ? 'b0 :
				     ((hold2_tag == 2'b10) && (hold2_add_valid == 5'b1)) ? 'b0 :
				     ((hold2_tag == 2'b10) && (hold2_shift_valid == 5'b1)) ? 'b1 :
				     cmd_block_table1[6*62 + 61];
				     
				     
	// tag 11
				     
	cmd_block_table1[7*62+0] <= (reset) ? 1'b0 :
				    ( (hold2_tag == 2'b11) && ( (hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 1'b1 :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     (( ~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table1[7*62+0];
				     
	cmd_block_table1[7*62+1] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[4*62+0] &&  block_condition &&
				    (!(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]))  && 
				    (!(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold2_data1 == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_result == cmd_block_table1[4*62+25:4*62+28]) && cmd_block_table1[4*62+24]) || 
				     ((hold2_result == cmd_block_table1[4*62+15:4*62+18]) && cmd_block_table1[4*62+14]) || 
				     ((hold2_result == cmd_block_table1[4*62+20:4*62+23]) && cmd_block_table1[4*62+19]) || 
				     (branch2 == 5'b10100) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[4*62+5:4*62+9] == 5'b10111) && branch2_cmd)) :
				    (((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) ||
				     ((~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table1[7*62+1];

	cmd_block_table1[7*62+2] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[5*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_result == cmd_block_table1[5*62+25:5*62+28]) && cmd_block_table1[5*62+24]) || 
				     ((hold2_result == cmd_block_table1[5*62+15:5*62+18]) && cmd_block_table1[5*62+14]) || 
				     ((hold2_result == cmd_block_table1[5*62+20:5*62+23]) && cmd_block_table1[5*62+19]) || 
				     (branch2 == 5'b10101) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[5*62+5:5*62+9] == 5'b10111) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[7*62+2];

	cmd_block_table1[7*62+3] <= (reset) ? 1'b0 :
				    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ? cmd_block_table1[2*62+0] && 
				    !(~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold2_data1 == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_data2 == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_result == cmd_block_table1[6*62+25:6*62+28]) && cmd_block_table1[6*62+24]) || 
				     ((hold2_result == cmd_block_table1[6*62+15:6*62+18]) && cmd_block_table1[6*62+14]) || 
				     ((hold2_result == cmd_block_table1[6*62+20:6*62+23]) && cmd_block_table1[6*62+19]) || 
				     (branch2 == 5'b10110) || 
				     (((hold2_cmd == 'b1100) || hold2_cmd == 'b1101) && (cmd_block_table1[6*62+5:6*62+9] == 5'b10111) && branch2_cmd)) :
				    ((~add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (~shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table1[7*62+3];

	cmd_block_table1[7*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table1[7*62+14:7*62+18]) && cmd_block_table1[7*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table1[7*62+19:7*62+23]) && cmd_block_table1[7*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[7*62+14:7*62+18]) && cmd_block_table1[7*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table1[7*62+19:7*62+23]) && cmd_block_table1[7*62+19]) ||
				     rw_case7) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table1[7*62+14:7*62+18]) || ~cmd_block_table1[7*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table1[7*62+19:7*62+23]) || ~cmd_block_table1[7*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[7*62+14:7*62+18]) || ~cmd_block_table1[7*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table1[7*62+19:7*62+23]) || ~cmd_block_table1[7*62+19]) &&
				   ~rw_case7) ? 1'b0 :
				    cmd_block_table1[7*62+4];
	
	cmd_block_table1[7*62+5:7*62+9] <= (reset) ? 5'b0 :
					  ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? branch2 :
					  cmd_block_table1[7*62+5:7*62+9];

	cmd_block_table1[7*62+10:7*62+13] <= (reset) ? 4'b0 :
					    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? hold2_cmd :
					    cmd_block_table1[7*62+10:7*62+13];

	cmd_block_table1[7*62+14:7*62+18] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b11) && (hold2_cmd != 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data1} :
					    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table1[7*62+14:7*62+18];

	cmd_block_table1[7*62+19:7*62+23] <= (reset) ? 5'b0 :
					   ((hold2_tag == 2'b11) && (hold2_cmd != 'b1001) && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_data2} :
					    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table1[7*62+19:7*62+23];

	cmd_block_table1[7*62+24:7*62+28] <= (reset) ? 5'b0 :
					    ((hold2_tag == 2'b11) && (hold2_cmd!= 'b1010) && (hold2_cmd != 'b1100) && (hold2_cmd != 'b1101) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? {1'b1, hold2_result} :
					    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b00001) || (hold2_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table1[7*62+24:7*62+28];

	cmd_block_table1[7*62+29:7*62+60] <= (reset) ? 32'b0 :
					    ((hold2_tag == 2'b11) && (hold2_cmd == 'b1001) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? hold2_data :
					    ((hold2_tag == 2'b11) && ((hold2_add_valid == 5'b1 || hold2_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table1[7*62+29:7*62+60];

	cmd_block_table1[7*62+61] <= (reset) ? 'b0 :
				  ((hold2_tag == 2'b11) && (hold2_add_valid == 5'b1)) ? 'b0 :
				  ((hold2_tag == 2'b11) && (hold2_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table1[7*62 + 61];


	//     ----------------------------------
	//     -- cmd_block_table for requestor 3
                //     ----------------------------------

	//     -- tag 00

	cmd_block_table2[0*62+0] <= (reset) ? 1'b0 : 
				    ( (hold3_tag == 2'b00) && ( (hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     (( shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table2[0*62+0];

	cmd_block_table2[0*62+1] <= (reset) ? 1'b0 :
				   ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[1*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_result == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_result == cmd_block_table2[1*62+15:1*62+18]) && cmd_block_table2[1*62+14]) || 
				     ((hold3_result == cmd_block_table2[1*62+20:1*62+23]) && cmd_block_table2[1*62+19]) || 
				     (branch3 == 5'b11001) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[1*62+5:1*62+9] == 5'b11000) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				   cmd_block_table2[0*62+1];

	cmd_block_table2[0*62+2] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[2*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_result == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_result == cmd_block_table2[2*62+15:2*62+18]) && cmd_block_table2[2*62+14]) || 
				     ((hold3_result == cmd_block_table2[2*62+20:2*62+23]) && cmd_block_table2[2*62+19]) || 
				     (branch3 == 5'b11010) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[2*62+5:2*62+9] == 5'b11000) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				   cmd_block_table2[0*62+2];

	cmd_block_table2[0*62+3] <= (reset) ? 1'b0 :
				   ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[3*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_result == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_result == cmd_block_table2[3*62+15:3*62+18]) && cmd_block_table2[3*62+14]) || 
				     ((hold3_result == cmd_block_table2[3*62+20:3*62+23]) && cmd_block_table2[3*62+19]) || 
				     (branch3 == 5'b11011) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[3*62+5:3*62+9] == 5'b11000) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[0*62+3];

	cmd_block_table2[0*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table2[0*62+14:0*62+18]) && cmd_block_table2[0*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table2[0*62+19:0*62+23]) && cmd_block_table2[0*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[0*62+14:0*62+18]) && cmd_block_table2[0*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[0*62+19:0*62+23]) && cmd_block_table2[0*62+19]) ||
				     rw_case8) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table2[0*62+14:0*62+18]) || ~cmd_block_table2[0*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table2[0*62+19:0*62+23]) || ~cmd_block_table2[0*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[0*62+14:0*62+18]) || ~cmd_block_table2[0*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[0*62+19:0*62+23]) || ~cmd_block_table2[0*62+19]) &&
				     ~rw_case8) ? 1'b0 :
				    cmd_block_table2[0*62+4];
	
	cmd_block_table2[0*62+5:0*62+9] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? branch3 :
					   cmd_block_table2[0*62+5:0*62+9];
	
	cmd_block_table2[0*62+10:0*62+13] <= (reset) ? 4'b0 :
					     ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? hold3_cmd :
					     cmd_block_table2[0*62+10:0*62+13];

	cmd_block_table2[0*62+14:0*62+18] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b0) && (hold3_cmd != 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data1} :
					    ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[0*62+14:0*62+18];

	cmd_block_table2[0*62+19:0*62+23] <= (reset) ? 5'b0 :
					     ((hold3_tag == 2'b0) && (hold3_cmd != 'b1001) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data2} :
					     ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table2[0*62+19:0*62+23];

	cmd_block_table2[0*62+24:0*62+28] <= (reset) ? 5'b0 :
					    ((hold3_tag == 2'b0) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && (hold3_cmd != 'b1101) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_result} :
					    ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[0*62+24:0*62+28];

	cmd_block_table2[0*62+29:0*62+60] <= (reset) ? 32'b0 :
					    ((hold3_tag == 2'b0) && (hold3_cmd == 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? hold3_data :
					    ((hold3_tag == 2'b0) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[0*62+29:0*62+60];

	cmd_block_table2[0*62+61] <= (reset) ? 'b0 :
				  ((hold3_tag == 2'b0) && (hold3_add_valid == 5'b1)) ? 'b0 :
				  ((hold3_tag == 2'b0) && (hold3_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[0*62+61];

	// tag 01

	cmd_block_table2[1*62+0] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b1) && ( (hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     (( shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table2[1*62+0];

	cmd_block_table2[1*62+1] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[0*62+0] && 
				    (!(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) )  && 
				    (!(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold3_data1 == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_result == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_result == cmd_block_table2[0*62+15:0*62+18]) && cmd_block_table2[0*62+14]) || 
				     ((hold3_result == cmd_block_table2[0*62+20:0*62+23]) && cmd_block_table2[0*62+19]) || 
				     (branch3 == 5'b11000) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[0*62+5:0*62+9] == 5'b11001) && branch3_cmd)) :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) ) || 
				     ((shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table2[1*62+1];

	cmd_block_table2[1*62+2] <= (reset) ? 1'b0 :
				   ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[2*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_result == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_result == cmd_block_table2[2*62+15:2*62+18]) && cmd_block_table2[2*62+14]) || 
				     ((hold3_result == cmd_block_table2[2*62+20:2*62+23]) && cmd_block_table2[2*62+19]) || 
				     (branch3 == 5'b11010) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[2*62+5:2*62+9] == 5'b11001) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[1*62+2];

	cmd_block_table2[1*62+3] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[3*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_result == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_result == cmd_block_table2[3*62+15:3*62+18]) && cmd_block_table2[3*62+14]) || 
				     ((hold3_result == cmd_block_table2[3*62+20:3*62+23]) && cmd_block_table2[3*62+19]) || 
				     (branch3 == 5'b11011) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[3*62+5:3*62+9] == 5'b11001) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[1*62+3];

	cmd_block_table2[1*62+4] <= (reset) ? 1'b0 :
				  (((rd_wr_conflict_block1 == cmd_block_table2[1*62+14:1*62+18]) && cmd_block_table2[1*62+14]) ||
				   ((rd_wr_conflict_block1 == cmd_block_table2[1*62+19:1*62+23]) && cmd_block_table2[1*62+19]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table2[1*62+14:1*62+18]) && cmd_block_table2[1*62+14]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table2[1*62+19:1*62+23]) && cmd_block_table2[1*62+19]) ||
				   rw_case9) ?  1'b1 :
				  (((rd_wr_conflict_block1 != cmd_block_table2[1*62+14:1*62+18]) || ~cmd_block_table2[1*62+14]) &&
				   ((rd_wr_conflict_block1 != cmd_block_table2[1*62+19:1*62+23]) || ~cmd_block_table2[1*62+19]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table2[1*62+14:1*62+18]) || ~cmd_block_table2[1*62+14]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table2[1*62+19:1*62+23]) || ~cmd_block_table2[1*62+19]) &&
				   ~rw_case9) ? 1'b0 :
				  cmd_block_table2[1*62+4];

	cmd_block_table2[1*62+5:1*62+9] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? branch3 :
					   cmd_block_table2[1*62+5:1*62+9];

	cmd_block_table2[1*62+10:1*62+13] <= (reset) ? 4'b0 :
					    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? hold3_cmd :
					    cmd_block_table2[1*62+10:1*62+13];

	cmd_block_table2[1*62+14:1*62+18] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b1) && (hold3_cmd != 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data1} :
					    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[1*62+14:1*62+18];

	cmd_block_table2[1*62+19:1*62+23] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b1) && (hold3_cmd != 'b1001) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data2} :
					    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[1*62+19:1*62+23];

	cmd_block_table2[1*62+24:1*62+28] <= (reset) ? 5'b0 :
					    ((hold3_tag == 2'b1) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && (hold3_cmd != 'b1101) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_result} :
					    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[1*62+24:1*62+28];

	cmd_block_table2[1*62+29:1*62+60] <= (reset) ? 32'b0 :
					    ((hold3_tag == 2'b1) && (hold3_cmd == 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? hold3_data :
					    ((hold3_tag == 2'b1) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[1*62+29:1*62+60];

	cmd_block_table2[1*62+61] <= (reset) ? 'b0 :
				  ((hold3_tag == 2'b1) && (hold3_add_valid == 5'b1)) ? 'b0 :
				  ((hold3_tag == 2'b1) && (hold3_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[1*62 + 61];
	
	// tag 10
	
	cmd_block_table2[2*62+0] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b10) && ( (hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 1'b1 :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) )  ||  
				     (( shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ) )) ? 
				    1'b0:
				    cmd_block_table2[2*62+0];

	cmd_block_table2[2*62+1] <= (reset) ? 1'b0 :
				   ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[0*62+0] && 
				    (!(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) && 
				    (!(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold3_data1 == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_result == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_result == cmd_block_table2[0*62+15:0*62+18]) && cmd_block_table2[0*62+14]) || 
				     ((hold3_result == cmd_block_table2[0*62+20:0*62+23]) && cmd_block_table2[0*62+19]) || 
				     (branch3 == 5'b11000) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[0*62+5:0*62+9] == 5'b11010) && branch3_cmd)) :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     ((shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table2[2*62+1];

	cmd_block_table2[2*62+2] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[1*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_result == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_result == cmd_block_table2[1*62+15:1*62+18]) && cmd_block_table2[1*62+14]) || 
				     ((hold3_result == cmd_block_table2[1*62+20:1*62+23]) && cmd_block_table2[1*62+19]) || 
				     (branch3 == 5'b11001) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[1*62+5:1*62+9] == 5'b11010) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[2*62+2];

	cmd_block_table2[2*62+3] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[3*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_result == cmd_block_table2[3*62+25:3*62+28]) && cmd_block_table2[3*62+24]) || 
				     ((hold3_result == cmd_block_table2[3*62+15:3*62+18]) && cmd_block_table2[3*62+14]) || 
				     ((hold3_result == cmd_block_table2[3*62+20:3*62+23]) && cmd_block_table2[3*62+19]) || 
				     (branch3 == 5'b11011) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[3*62+5:3*62+9] == 5'b11010) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[2*62+3];

	cmd_block_table2[2*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table2[2*62+14:2*62+18]) && cmd_block_table2[2*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table2[2*62+19:2*62+23]) && cmd_block_table2[2*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[2*62+14:2*62+18]) && cmd_block_table2[2*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[2*62+19:2*62+23]) && cmd_block_table2[2*62+19]) ||
				     rw_case10) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table2[2*62+14:2*62+18]) || ~cmd_block_table2[2*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table2[2*62+19:2*62+23]) || ~cmd_block_table2[2*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[2*62+14:2*62+18]) || ~cmd_block_table2[2*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[2*62+19:2*62+23]) || ~cmd_block_table2[2*62+19]) &&
				     ~rw_case10) ? 1'b0 :
				    cmd_block_table2[2*62+4];

	cmd_block_table2[2*62+5:2*62+9] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? branch3 :
					   cmd_block_table2[2*62+5:2*62+9];
	
	cmd_block_table2[2*62+10:2*62+13] <= (reset) ? 4'b0 :
					     ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? hold3_cmd :
					     cmd_block_table2[2*62+10:2*62+13];
	
	cmd_block_table2[2*62+14:2*62+18] <= (reset) ? 5'b0 :
					     ((hold3_tag == 2'b10) && (hold3_cmd != 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data1} :
					     ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table2[2*62+14:2*62+18];
	
	cmd_block_table2[2*62+19:2*62+23] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b10) && (hold3_cmd != 'b1001) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data2} :
					    ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[2*62+19:2*62+23];

	cmd_block_table2[2*62+24:2*62+28] <= (reset) ? 5'b0 :
					    ((hold3_tag == 2'b10) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && (hold3_cmd != 'b1101) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_result} :
					    ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[2*62+24:2*62+28];

	cmd_block_table2[2*62+29:2*62+60] <= (reset) ? 32'b0 :
					    ((hold3_tag == 2'b10) && (hold3_cmd == 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? hold3_data :
					    ((hold3_tag == 2'b10) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[2*62+29:2*62+60];

	cmd_block_table2[2*62+61] <= (reset) ? 'b0 :
				  ((hold3_tag == 2'b10) && (hold3_add_valid == 5'b1)) ? 'b0 :
				  ((hold3_tag == 2'b10) && (hold3_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[2*62 + 61];


	// tag 11

	cmd_block_table2[3*62+0] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b11) && ( (hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 1'b1 :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     ((shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table2[3*62+0];

	cmd_block_table2[3*62+1] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[0*62+0] && 
				    (!(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) )  && 
				    (!(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold3_data1 == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_result == cmd_block_table2[0*62+25:0*62+28]) && cmd_block_table2[0*62+24]) || 
				     ((hold3_result == cmd_block_table2[0*62+15:0*62+18]) && cmd_block_table2[0*62+14]) || 
				     ((hold3_result == cmd_block_table2[0*62+20:0*62+23]) && cmd_block_table2[0*62+19]) || 
				     (branch3 == 5'b11000) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[0*62+5:0*62+9] == 5'b11011) && branch3_cmd)) :
				    (((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) ||
				     ((shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table2[3*62+1];

	cmd_block_table2[3*62+2] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[1*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_result == cmd_block_table2[1*62+25:1*62+28]) && cmd_block_table2[1*62+24]) || 
				     ((hold3_result == cmd_block_table2[1*62+15:1*62+18]) && cmd_block_table2[1*62+14]) || 
				     ((hold3_result == cmd_block_table2[1*62+20:1*62+23]) && cmd_block_table2[1*62+19]) || 
				     (branch3 == 5'b11001) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[1*62+5:1*62+9] == 5'b11011) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[3*62+2];

	cmd_block_table2[3*62+3] <= (reset) ? 1'b0 :
				    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ? cmd_block_table2[2*62+0] && 
				    !(add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold3_data1 == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_data2 == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_result == cmd_block_table2[2*62+25:2*62+28]) && cmd_block_table2[2*62+24]) || 
				     ((hold3_result == cmd_block_table2[2*62+15:2*62+18]) && cmd_block_table2[2*62+14]) || 
				     ((hold3_result == cmd_block_table2[2*62+20:2*62+23]) && cmd_block_table2[2*62+19]) || 
				     (branch3 == 5'b11010) || 
				     (((hold3_cmd == 'b1100) || hold3_cmd == 'b1101) && (cmd_block_table2[2*62+5:2*62+9] == 5'b11011) && branch3_cmd)) :
				    ((add_queue[dispatch_add*4] && ~add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && ~shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[3*62+3];

	cmd_block_table2[3*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table2[3*62+14:3*62+18]) && cmd_block_table2[3*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table2[3*62+19:3*62+23]) && cmd_block_table2[3*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[3*62+14:3*62+18]) && cmd_block_table2[3*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[3*62+19:3*62+23]) && cmd_block_table2[3*62+19]) ||
				     rw_case11) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table2[3*62+14:3*62+18]) || ~cmd_block_table2[3*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table2[3*62+19:3*62+23]) || ~cmd_block_table2[3*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[3*62+14:3*62+18]) || ~cmd_block_table2[3*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[3*62+19:3*62+23]) || ~cmd_block_table2[3*62+19]) &&
				   ~rw_case11) ? 1'b0 :
				    cmd_block_table2[3*62+4];
	
	cmd_block_table2[3*62+5:3*62+9] <= (reset) ? 5'b0 :
					  ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? branch3 :
					  cmd_block_table2[3*62+5:3*62+9];

	cmd_block_table2[3*62+10:3*62+13] <= (reset) ? 4'b0 :
					    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? hold3_cmd :
					    cmd_block_table2[3*62+10:3*62+13];

	cmd_block_table2[3*62+14:3*62+18] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b11) && (hold3_cmd != 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data1} :
					    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[3*62+14:3*62+18];

	cmd_block_table2[3*62+19:3*62+23] <= (reset) ? 5'b0 :
					   ((hold3_tag == 2'b11) && (hold3_cmd != 'b1001) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_data2} :
					    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[3*62+19:3*62+23];

	cmd_block_table2[3*62+24:3*62+28] <= (reset) ? 5'b0 :
					    ((hold3_tag == 2'b11) && (hold3_cmd!= 'b1010) && (hold3_cmd != 'b1100) && (hold3_cmd != 'b1101) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? {1'b1, hold3_result} :
					    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b00001) || (hold3_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[3*62+24:3*62+28];

	cmd_block_table2[3*62+29:3*62+60] <= (reset) ? 32'b0 :
					    ((hold3_tag == 2'b11) && (hold3_cmd == 'b1001) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? hold3_data :
					    ((hold3_tag == 2'b11) && ((hold3_add_valid == 5'b1 || hold3_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[3*62+29:3*62+60];

	cmd_block_table2[3*62+61] <= (reset) ? 'b0 :
				  ((hold3_tag == 2'b11) && (hold3_add_valid == 5'b1)) ? 'b0 :
				  ((hold3_tag == 2'b11) && (hold3_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[3*62 + 61];


//	 ----------------------------------
//	 -- cmd_block_table for requestor 4
//	 ----------------------------------

//	 -- tag 00
	 
	cmd_block_table2[4*62+0] <= (reset) ? 1'b0 : 
				    ( (hold4_tag == 2'b00) && ( (hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     (( shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table2[4*62+0];

	cmd_block_table2[4*62+1] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[5*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_result == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_result == cmd_block_table2[5*62+15:5*62+18]) && cmd_block_table2[5*62+14]) || 
				     ((hold4_result == cmd_block_table2[5*62+20:5*62+23]) && cmd_block_table2[5*62+19]) || 
				     (branch4 == 5'b11101) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[5*62+5:5*62+9] == 5'b11100) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[4*62+1];

	cmd_block_table2[4*62+2] <= (reset) ? 1'b0 :
				   ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[6*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_result == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_result == cmd_block_table2[6*62+15:6*62+18]) && cmd_block_table2[6*62+14]) || 
				     ((hold4_result == cmd_block_table2[6*62+20:6*62+23]) && cmd_block_table2[6*62+19]) || 
				     (branch4 == 5'b11110) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[6*62+5:6*62+9] == 5'b11100) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				   cmd_block_table2[4*62+2];

	cmd_block_table2[4*62+3] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[7*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_result == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_result == cmd_block_table2[7*62+15:7*62+18]) && cmd_block_table2[7*62+14]) || 
				     ((hold4_result == cmd_block_table2[7*62+20:7*62+23]) && cmd_block_table2[7*62+19]) || 
				     (branch4 == 5'b11111) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[7*62+5:7*62+9] == 5'b11100) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[4*62+3];

	cmd_block_table2[4*62+4] <= (reset) ? 1'b0 :
				  (((rd_wr_conflict_block1 == cmd_block_table2[4*62+14:4*62+18]) && cmd_block_table2[4*62+14]) ||
				   ((rd_wr_conflict_block1 == cmd_block_table2[4*62+19:4*62+23]) && cmd_block_table2[4*62+19]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table2[4*62+14:4*62+18]) && cmd_block_table2[4*62+14]) ||
				   ((rd_wr_conflict_block2 == cmd_block_table2[4*62+19:4*62+23]) && cmd_block_table2[4*62+19]) ||
				   rw_case12) ?  1'b1 :
				  (((rd_wr_conflict_block1 != cmd_block_table2[4*62+14:4*62+18]) || ~cmd_block_table2[4*62+14]) &&
				   ((rd_wr_conflict_block1 != cmd_block_table2[4*62+19:4*62+23]) || ~cmd_block_table2[4*62+19]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table2[4*62+14:4*62+18]) || ~cmd_block_table2[4*62+14]) &&
				   ((rd_wr_conflict_block2 != cmd_block_table2[4*62+19:4*62+23]) || ~cmd_block_table2[4*62+19]) &&
				   ~rw_case12) ? 1'b0 :
				  cmd_block_table2[4*62+4];

	cmd_block_table2[4*62+5:4*62+9] <= (reset) ? 5'b0 :
					  ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? branch4 :
					  cmd_block_table2[4*62+5:4*62+9];

	cmd_block_table2[4*62+10:4*62+13] <= (reset) ? 4'b0 :
					    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? hold4_cmd :
					    cmd_block_table2[4*62+10:4*62+13];

	cmd_block_table2[4*62+14:4*62+18] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b0) && (hold4_cmd != 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data1} :
					    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[4*62+14:4*62+18];

	cmd_block_table2[4*62+19:4*62+23] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b0) && (hold4_cmd != 'b1001) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data2} :
					    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[4*62+19:4*62+23];

	cmd_block_table2[4*62+24:4*62+28] <= (reset) ? 5'b0 :
					    ((hold4_tag == 2'b0) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && (hold4_cmd != 'b1101) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_result} :
					    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[4*62+24:4*62+28];

	cmd_block_table2[4*62+29:4*62+60] <= (reset) ? 32'b0 :
					    ((hold4_tag == 2'b0) && (hold4_cmd == 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? hold4_data :
					    ((hold4_tag == 2'b0) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[4*62+29:4*62+60];

	cmd_block_table2[4*62+61] <= (reset) ? 'b0 :
				  ((hold4_tag == 2'b0) && (hold4_add_valid == 5'b1)) ? 'b0 :
				  ((hold4_tag == 2'b0) && (hold4_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[4*62+61];

	// tag 01

	cmd_block_table2[5*62+0] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b1) && ( (hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 
				    1'b1 :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     (( shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table2[5*62+0];

	cmd_block_table2[5*62+1] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[4*62+0] && 
				    (!(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]))  && 
				    (!(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold4_data1 == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_result == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_result == cmd_block_table2[4*62+15:4*62+18]) && cmd_block_table2[4*62+14]) || 
				     ((hold4_result == cmd_block_table2[4*62+20:4*62+23]) && cmd_block_table2[4*62+19]) || 
				     (branch4 == 5'b11100) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[4*62+5:4*62+9] == 5'b11101) && branch4_cmd)) :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     ((shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table2[5*62+1];

	cmd_block_table2[5*62+2] <= (reset) ? 1'b0 : ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[6*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_result == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_result == cmd_block_table2[6*62+15:6*62+18]) && cmd_block_table2[6*62+14]) || 
				     ((hold4_result == cmd_block_table2[6*62+20:6*62+23]) && cmd_block_table2[6*62+19]) || 
				     (branch4 == 5'b11110) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[6*62+5:6*62+9] == 5'b11101) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[5*62+2];

	cmd_block_table2[5*62+3] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[7*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_result == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_result == cmd_block_table2[7*62+15:7*62+18]) && cmd_block_table2[7*62+14]) || 
				     ((hold4_result == cmd_block_table2[7*62+20:7*62+23]) && cmd_block_table2[7*62+19]) || 
				     (branch4 == 5'b11111) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[7*62+5:7*62+9] == 5'b11101) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[5*62+3];

	cmd_block_table2[5*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table2[5*62+14:5*62+18]) && cmd_block_table2[5*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table2[5*62+19:5*62+23]) && cmd_block_table2[5*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[5*62+14:5*62+18]) && cmd_block_table2[5*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[5*62+19:5*62+23]) && cmd_block_table2[5*62+19]) ||
				     rw_case13) ?  
				    1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table2[5*62+14:5*62+18]) || ~cmd_block_table2[5*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table2[5*62+19:5*62+23]) || ~cmd_block_table2[5*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[5*62+14:5*62+18]) || ~cmd_block_table2[5*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[5*62+19:5*62+23]) || ~cmd_block_table2[5*62+19]) &&
				     ~rw_case13) ? 
				    1'b0 :
				    cmd_block_table2[5*62+4];

	cmd_block_table2[5*62+5:5*62+9] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? branch4 :
					   cmd_block_table2[5*62+5:5*62+9];

	cmd_block_table2[5*62+10:5*62+13] <= (reset) ? 4'b0 :
					    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? hold4_cmd :
					    cmd_block_table2[5*62+10:5*62+13];

	cmd_block_table2[5*62+14:5*62+18] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b1) && (hold4_cmd != 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data1} :
					    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[5*62+14:5*62+18];

	cmd_block_table2[5*62+19:5*62+23] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b1) && (hold4_cmd != 'b1001) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data2} :
					    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[5*62+19:5*62+23];

	cmd_block_table2[5*62+24:5*62+28] <= (reset) ? 5'b0 :
					    ((hold4_tag == 2'b1) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && (hold4_cmd != 'b1101) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_result} :
					    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[5*62+24:5*62+28];

	cmd_block_table2[5*62+29:5*62+60] <= (reset) ? 32'b0 :
					    ((hold4_tag == 2'b1) && (hold4_cmd == 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? hold4_data :
					    ((hold4_tag == 2'b1) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[5*62+29:5*62+60];

	cmd_block_table2[5*62+61] <= (reset) ? 'b0 :
				  ((hold4_tag == 2'b1) && (hold4_add_valid == 5'b1)) ? 'b0 :
				  ((hold4_tag == 2'b1) && (hold4_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[5*62 + 61];
	
	// tag 10
	
	cmd_block_table2[6*62+0] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b10) && ( (hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 1'b1 :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) )  ||  
				     (( shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3] ) )) ? 
				    1'b0:
				    cmd_block_table2[6*62+0];

	cmd_block_table2[6*62+1] <= (reset) ? 1'b0 :
				   ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[4*62+0] && 
				    (!(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) && 
				    (!(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold4_data1 == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_result == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_result == cmd_block_table2[4*62+15:4*62+18]) && cmd_block_table2[4*62+14]) || 
				     ((hold4_result == cmd_block_table2[4*62+20:4*62+23]) && cmd_block_table2[4*62+19]) || 
				     (branch4 == 5'b11100) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[4*62+5:4*62+9] == 5'b11110) && branch4_cmd)) :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) || 
				     ((shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])))? 
				    1'b0 :
				    cmd_block_table2[6*62+1];
	
	cmd_block_table2[6*62+2] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[5*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_result == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_result == cmd_block_table2[5*62+15:5*62+18]) && cmd_block_table2[5*62+14]) || 
				     ((hold4_result == cmd_block_table2[5*62+20:5*62+23]) && cmd_block_table2[5*62+19]) || 
				     (branch4 == 5'b11101) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[5*62+5:5*62+9] == 5'b11110) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[6*62+2];
				     
	cmd_block_table2[6*62+3] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[7*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_result == cmd_block_table2[7*62+25:7*62+28]) && cmd_block_table2[7*62+24]) || 
				     ((hold4_result == cmd_block_table2[7*62+15:7*62+18]) && cmd_block_table2[7*62+14]) || 
				     ((hold4_result == cmd_block_table2[7*62+20:7*62+23]) && cmd_block_table2[7*62+19]) || 
				     (branch4 == 5'b11111) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[7*62+5:7*62+9] == 5'b11110) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[6*62+3];

	cmd_block_table2[6*62+4] <= (reset) ? 1'b0 :
				     (((rd_wr_conflict_block1 == cmd_block_table2[6*62+14:6*62+18]) && cmd_block_table2[6*62+14]) ||
				      ((rd_wr_conflict_block1 == cmd_block_table2[6*62+19:6*62+23]) && cmd_block_table2[6*62+19]) ||
				      ((rd_wr_conflict_block2 == cmd_block_table2[6*62+14:6*62+18]) && cmd_block_table2[6*62+14]) ||
				      ((rd_wr_conflict_block2 == cmd_block_table2[6*62+19:6*62+23]) && cmd_block_table2[6*62+19]) ||
				      rw_case14) ?  1'b1 :
				     (((rd_wr_conflict_block1 != cmd_block_table2[6*62+14:6*62+18]) || ~cmd_block_table2[6*62+14]) &&
				      ((rd_wr_conflict_block1 != cmd_block_table2[6*62+19:6*62+23]) || ~cmd_block_table2[6*62+19]) &&
				      ((rd_wr_conflict_block2 != cmd_block_table2[6*62+14:6*62+18]) || ~cmd_block_table2[6*62+14]) &&
				      ((rd_wr_conflict_block2 != cmd_block_table2[6*62+19:6*62+23]) || ~cmd_block_table2[6*62+19]) &&
				      ~rw_case14) ? 1'b0 :
				    cmd_block_table2[6*62+4];
	
	cmd_block_table2[6*62+5:6*62+9] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? branch4 :
					   cmd_block_table2[6*62+5:6*62+9];
	
	cmd_block_table2[6*62+10:6*62+13] <= (reset) ? 4'b0 :
					     ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? hold4_cmd :
					     cmd_block_table2[6*62+10:6*62+13];
	
	cmd_block_table2[6*62+14:6*62+18] <= (reset) ? 5'b0 :
					     ((hold4_tag == 2'b10) && (hold4_cmd != 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data1} :
					     ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table2[6*62+14:6*62+18];
	
	cmd_block_table2[6*62+19:6*62+23] <= (reset) ? 5'b0 :
					     ((hold4_tag == 2'b10) && (hold4_cmd != 'b1001) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data2} :
					     ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table2[6*62+19:6*62+23];
	
	cmd_block_table2[6*62+24:6*62+28] <= (reset) ? 5'b0 :
					     ((hold4_tag == 2'b10) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && (hold4_cmd != 'b1101) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_result} :
					     ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					     cmd_block_table2[6*62+24:6*62+28];
	
	cmd_block_table2[6*62+29:6*62+60] <= (reset) ? 32'b0 :
					     ((hold4_tag == 2'b10) && (hold4_cmd == 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? hold4_data :
					     ((hold4_tag == 2'b10) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? 32'b0 :
					     cmd_block_table2[6*62+29:6*62+60];
	
	cmd_block_table2[6*62+61] <= (reset) ? 'b0 :
				     ((hold4_tag == 2'b10) && (hold4_add_valid == 5'b1)) ? 'b0 :
				     ((hold4_tag == 2'b10) && (hold4_shift_valid == 5'b1)) ? 'b1 :
				     cmd_block_table2[6*62 + 61];
				     
				     
	// tag 11
				     
	cmd_block_table2[7*62+0] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b11) && ( (hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 1'b1 :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3])) || 
				     ((shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3] ))) ? 
				    1'b0:
				    cmd_block_table2[7*62+0];
				     
	cmd_block_table2[7*62+1] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[4*62+0] &&
				    (!(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]))  && 
				    (!(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) && 
				    (((hold4_data1 == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_result == cmd_block_table2[4*62+25:4*62+28]) && cmd_block_table2[4*62+24]) || 
				     ((hold4_result == cmd_block_table2[4*62+15:4*62+18]) && cmd_block_table2[4*62+14]) || 
				     ((hold4_result == cmd_block_table2[4*62+20:4*62+23]) && cmd_block_table2[4*62+19]) || 
				     (branch4 == 5'b11100) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[4*62+5:4*62+9] == 5'b11111) && branch4_cmd)) :
				    (((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3])) ||
				     ((shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]))) ? 
				    1'b0 :
				    cmd_block_table2[7*62+1];

	cmd_block_table2[7*62+2] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[5*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_result == cmd_block_table2[5*62+25:5*62+28]) && cmd_block_table2[5*62+24]) || 
				     ((hold4_result == cmd_block_table2[5*62+15:5*62+18]) && cmd_block_table2[5*62+14]) || 
				     ((hold4_result == cmd_block_table2[5*62+20:5*62+23]) && cmd_block_table2[5*62+19]) || 
				     (branch4 == 5'b11101) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[5*62+5:5*62+9] == 5'b11111) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && ~add_queue[dispatch_add*4+2] && add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && ~shift_queue[dispatch_shift*4+2] && shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[7*62+2];

	cmd_block_table2[7*62+3] <= (reset) ? 1'b0 :
				    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ? cmd_block_table2[6*62+0] && 
				    !(add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) &&  
				    !(shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3]) && 
				    (((hold4_data1 == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_data2 == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_result == cmd_block_table2[6*62+25:6*62+28]) && cmd_block_table2[6*62+24]) || 
				     ((hold4_result == cmd_block_table2[6*62+15:6*62+18]) && cmd_block_table2[6*62+14]) || 
				     ((hold4_result == cmd_block_table2[6*62+20:6*62+23]) && cmd_block_table2[6*62+19]) || 
				     (branch4 == 5'b11110) || 
				     (((hold4_cmd == 'b1100) || hold4_cmd == 'b1101) && (cmd_block_table2[6*62+5:6*62+9] == 5'b11111) && branch4_cmd)) :
				    ((add_queue[dispatch_add*4] && add_queue[dispatch_add*4+1] && add_queue[dispatch_add*4+2] && ~add_queue[dispatch_add*4+3]) || 
				     (shift_queue[dispatch_shift*4] && shift_queue[dispatch_shift*4+1] && shift_queue[dispatch_shift*4+2] && ~shift_queue[dispatch_shift*4+3])) ? 
				    1'b0 :
				    cmd_block_table2[7*62+3];

	cmd_block_table2[7*62+4] <= (reset) ? 1'b0 :
				    (((rd_wr_conflict_block1 == cmd_block_table2[7*62+14:7*62+18]) && cmd_block_table2[7*62+14]) ||
				     ((rd_wr_conflict_block1 == cmd_block_table2[7*62+19:7*62+23]) && cmd_block_table2[7*62+19]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[7*62+14:7*62+18]) && cmd_block_table2[7*62+14]) ||
				     ((rd_wr_conflict_block2 == cmd_block_table2[7*62+19:7*62+23]) && cmd_block_table2[7*62+19]) ||
				     rw_case15) ?  1'b1 :
				    (((rd_wr_conflict_block1 != cmd_block_table2[7*62+14:7*62+18]) || ~cmd_block_table2[7*62+14]) &&
				     ((rd_wr_conflict_block1 != cmd_block_table2[7*62+19:7*62+23]) || ~cmd_block_table2[7*62+19]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[7*62+14:7*62+18]) || ~cmd_block_table2[7*62+14]) &&
				     ((rd_wr_conflict_block2 != cmd_block_table2[7*62+19:7*62+23]) || ~cmd_block_table2[7*62+19]) &&
				   ~rw_case15) ? 1'b0 :
				    cmd_block_table2[7*62+4];
	
	cmd_block_table2[7*62+5:7*62+9] <= (reset) ? 5'b0 :
					  ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? branch4 :
					  cmd_block_table2[7*62+5:7*62+9];

	cmd_block_table2[7*62+10:7*62+13] <= (reset) ? 4'b0 :
					    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? hold4_cmd :
					    cmd_block_table2[7*62+10:7*62+13];

	cmd_block_table2[7*62+14:7*62+18] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b11) && (hold4_cmd != 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data1} :
					    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[7*62+14:7*62+18];

	cmd_block_table2[7*62+19:7*62+23] <= (reset) ? 5'b0 :
					   ((hold4_tag == 2'b11) && (hold4_cmd != 'b1001) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_data2} :
					    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					   cmd_block_table2[7*62+19:7*62+23];

	cmd_block_table2[7*62+24:7*62+28] <= (reset) ? 5'b0 :
					     ((hold4_tag == 2'b11) && (hold4_cmd!= 'b1010) && (hold4_cmd != 'b1100) && (hold4_cmd != 'b1101) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? {1'b1, hold4_result} :
					    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b00001) || (hold4_shift_valid == 5'b00001))) ? 5'b0 :
					    cmd_block_table2[7*62+24:7*62+28];

	cmd_block_table2[7*62+29:7*62+60] <= (reset) ? 32'b0 :
					    ((hold4_tag == 2'b11) && (hold4_cmd == 'b1001) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? hold4_data :
					    ((hold4_tag == 2'b11) && ((hold4_add_valid == 5'b1 || hold4_shift_valid == 5'b1))) ? 32'b0 :
					    cmd_block_table2[7*62+29:7*62+60];

	cmd_block_table2[7*62+61] <= (reset) ? 'b0 :
				  ((hold4_tag == 2'b11) && (hold4_add_valid == 5'b1)) ? 'b0 :
				  ((hold4_tag == 2'b11) && (hold4_shift_valid == 5'b1)) ? 'b1 :
				  cmd_block_table2[7*62 + 61];

	
	
     end // always @ (negedge c_clk)
   
   assign skip_11 = 'b1;
   assign block_condition = 'b0;

   // holdx_add_valid and holdx_shift_valid is set to 1 if there's a valid new command from holdreg x.

   assign hold1_add_valid = ((hold1_cmd == 'b0001) || (hold1_cmd == 'b0010) || (hold1_cmd == 'b1100) || (hold1_cmd == 'b1101)) ? 'b00001 :
	  5'b0;

   assign hold2_add_valid = ((hold2_cmd == 'b0001) || (hold2_cmd == 'b0010) || (hold2_cmd == 'b1100) || (hold2_cmd == 'b1101)) ? 'b00001 :
	  5'b0;

   assign hold3_add_valid = ((hold3_cmd == 'b0001) || (hold3_cmd == 'b0010) || (hold3_cmd == 'b1100) || (hold3_cmd == 'b1101)) ? 'b00001 :
	  5'b0;

   assign hold4_add_valid = ((hold4_cmd == 'b0001) || (hold4_cmd == 'b0010) || (hold4_cmd == 'b1100) || (hold4_cmd == 'b1101)) ? 'b00001 :
	  5'b0;

   assign hold1_shift_valid = ((hold1_cmd == 'b0101) || (hold1_cmd == 'b0110) || (hold1_cmd == 'b1001) || (hold1_cmd == 'b1010)) ? 'b00001 :
	  5'b0;

   assign hold2_shift_valid = ((hold2_cmd == 'b0101) || (hold2_cmd == 'b0110) || (hold2_cmd == 'b1001) || (hold2_cmd == 'b1010)) ? 'b00001 :
	  5'b0;

   assign hold3_shift_valid = ((hold3_cmd == 'b0101) || (hold3_cmd == 'b0110) || (hold3_cmd == 'b1001) || (hold3_cmd == 'b1010)) ? 'b00001 :
	  5'b0;

   assign hold4_shift_valid = ((hold4_cmd == 'b0101) || (hold4_cmd == 'b0110) || (hold4_cmd == 'b1001) || (hold4_cmd == 'b1010)) ? 'b00001 :
	  5'b0;

 

   // holdsx_add_pos is where the command and data willgo in the addqueu
   // a vaue of "00000" means there's no new command this cycle

   assign hold1_add_pos = (hold1_add_valid == 5'b0) ? 5'b0 :
	  ((addqueue_curpos == 5'b0) || (dispatch_add == 5'b0)) ? addqueue_curpos + 1 :
	  addqueue_curpos;

   assign hold2_add_pos = (hold2_add_valid == 5'b0) ? 5'b0 :
	  ((addqueue_curpos == 5'b0) || (dispatch_add == 5'b0)) ? addqueue_curpos + hold1_add_valid + 1 :
	  addqueue_curpos + hold1_add_valid;

   assign hold3_add_pos = (hold3_add_valid == 5'b0) ? 5'b0 :
	  ((addqueue_curpos == 5'b0) || (dispatch_add == 5'b0)) ? addqueue_curpos + hold1_add_valid + hold2_add_valid + 1 :
	  addqueue_curpos + hold1_add_valid + hold2_add_valid;

   assign hold4_add_pos = (hold4_add_valid == 5'b0) ? 5'b0 :
	  ((addqueue_curpos == 5'b0) || (dispatch_add == 5'b0)) ? addqueue_curpos + hold1_add_valid + hold2_add_valid + hold3_add_valid + 1 :
	  addqueue_curpos + hold1_add_valid + hold2_add_valid + hold3_add_valid;

   // holdx_shift_pos is where the command and data will go in the shiftqueue
   // a value of "00000" means there'sno new command this cycle

   assign hold1_shift_pos = (hold1_shift_valid == 5'b0) ? 5'b0 :
	  ((shiftqueue_curpos == 5'b0) || (dispatch_shift == 5'b0)) ? shiftqueue_curpos + 1 :
	  shiftqueue_curpos;

   assign hold2_shift_pos = (hold2_shift_valid == 5'b0) ? 5'b0 :
	  ((shiftqueue_curpos == 5'b0) || (dispatch_shift == 5'b0)) ? shiftqueue_curpos + hold1_shift_valid + 1 :
	  shiftqueue_curpos + hold1_shift_valid;

   assign hold3_shift_pos = (hold3_shift_valid == 5'b0) ? 5'b0 :
	  ((shiftqueue_curpos == 5'b0) || (dispatch_shift == 5'b0)) ? shiftqueue_curpos + hold1_shift_valid + hold2_shift_valid + 1 :
	  shiftqueue_curpos + hold1_shift_valid + hold2_shift_valid;

   assign hold4_shift_pos = (hold4_shift_valid == 5'b0) ? 5'b0 :
	  ((shiftqueue_curpos == 5'b0) || (dispatch_shift == 5'b0)) ? shiftqueue_curpos + hold1_shift_valid + hold2_shift_valid + hold3_shift_valid + 1 :
	  shiftqueue_curpos + hold1_shift_valid + hold2_shift_valid + hold3_shift_valid;

//     temp_dispatch_add is set to the index of oldest cmd in the add_queue that can be dispatched
//    (valid and not blocked by another cmd), it is set to "00000" if the add_queue is empty
//     or if every valid cmd in the add_queue is blocked

   assign temp_dispatch_add = (reset) ? 5'b0 :
	  
	  (( ~add_queue[1*4+0] &&
	     ( cmd_block_table1[ (add_queue[1*4+1:1*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[1*4+1:1*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[1*4+1:1*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[1*4+1:1*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[1*4+1:1*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[1*4+1:1*4+3])*62+61] )  ||
	   ( add_queue[1*4+0] &&
	     ( cmd_block_table2[ (add_queue[1*4+1:1*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[1*4+1:1*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[1*4+1:1*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[1*4+1:1*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[1*4+1:1*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[1*4+1:1*4+3])*62+61] )) ? 5'b1 :

	  (( ~add_queue[2*4+0] &&
	     ( cmd_block_table1[ (add_queue[2*4+1:2*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[2*4+1:2*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[2*4+1:2*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[2*4+1:2*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[2*4+1:2*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[2*4+1:2*4+3])*62+61] )  ||
	   ( add_queue[2*4+0] &&
	     ( cmd_block_table2[ (add_queue[2*4+1:2*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[2*4+1:2*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[2*4+1:2*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[2*4+1:2*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[2*4+1:2*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[2*4+1:2*4+3])*62+61] )) ? 5'b10 :

	  (( ~add_queue[3*4+0] &&
	     ( cmd_block_table1[ (add_queue[3*4+1:3*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[3*4+1:3*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[3*4+1:3*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[3*4+1:3*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[3*4+1:3*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[3*4+1:3*4+3])*62+61] )  ||
	   ( add_queue[3*4+0] &&
	     ( cmd_block_table2[ (add_queue[3*4+1:3*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[3*4+1:3*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[3*4+1:3*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[3*4+1:3*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[3*4+1:3*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[3*4+1:3*4+3])*62+61] )) ? 5'b11 :
	   
	  (( ~add_queue[4*4+0] &&
	     ( cmd_block_table1[ (add_queue[4*4+1:4*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[4*4+1:4*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[4*4+1:4*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[4*4+1:4*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[4*4+1:4*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[4*4+1:4*4+3])*62+61] )  ||
	   ( add_queue[4*4+0] &&
	     ( cmd_block_table2[ (add_queue[4*4+1:4*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[4*4+1:4*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[4*4+1:4*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[4*4+1:4*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[4*4+1:4*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[4*4+1:4*4+3])*62+61] )) ? 5'b100 :

	   (( ~add_queue[5*4+0] &&
	     ( cmd_block_table1[ (add_queue[5*4+1:5*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[5*4+1:5*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[5*4+1:5*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[5*4+1:5*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[5*4+1:5*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[5*4+1:5*4+3])*62+61] )  ||
	   ( add_queue[5*4+0] &&
	     ( cmd_block_table2[ (add_queue[5*4+1:5*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[5*4+1:5*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[5*4+1:5*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[5*4+1:5*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[5*4+1:5*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[5*4+1:5*4+3])*62+61] )) ? 5'b101 :

	  (( ~add_queue[6*4+0] &&
	     ( cmd_block_table1[ (add_queue[6*4+1:6*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[6*4+1:6*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[6*4+1:6*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[6*4+1:6*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[6*4+1:6*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[6*4+1:6*4+3])*62+61] )  ||
	   ( add_queue[6*4+0] &&
	     ( cmd_block_table2[ (add_queue[6*4+1:6*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[6*4+1:6*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[6*4+1:6*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[6*4+1:6*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[6*4+1:6*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[6*4+1:6*4+3])*62+61] )) ? 5'b110 :

	  (( ~add_queue[7*4+0] &&
	     ( cmd_block_table1[ (add_queue[7*4+1:7*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[7*4+1:7*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[7*4+1:7*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[7*4+1:7*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[7*4+1:7*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[7*4+1:7*4+3])*62+61] )  ||
	   ( add_queue[7*4+0] &&
	     ( cmd_block_table2[ (add_queue[7*4+1:7*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[7*4+1:7*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[7*4+1:7*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[7*4+1:7*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[7*4+1:7*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[7*4+1:7*4+3])*62+61] )) ? 5'b111 :

	   (( ~add_queue[8*4+0] && 
	     ( cmd_block_table1[ (add_queue[8*4+1:8*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[8*4+1:8*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[8*4+1:8*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[8*4+1:8*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[8*4+1:8*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[8*4+1:8*4+3])*62+61] )  ||
	   ( add_queue[8*4+0] &&
	     ( cmd_block_table2[ (add_queue[8*4+1:8*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[8*4+1:8*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[8*4+1:8*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[8*4+1:8*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[8*4+1:8*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[8*4+1:8*4+3])*62+61] )) ? 5'b1000 :

	  (( ~add_queue[9*4+0] &&
	     ( cmd_block_table1[ (add_queue[9*4+1:9*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[9*4+1:9*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[9*4+1:9*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[9*4+1:9*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[9*4+1:9*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[9*4+1:9*4+3])*62+61] )  ||
	   ( add_queue[9*4+0] &&
	     ( cmd_block_table2[ (add_queue[9*4+1:9*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[9*4+1:9*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[9*4+1:9*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[9*4+1:9*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[9*4+1:9*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[9*4+1:9*4+3])*62+61] )) ? 5'b1001 :

	  (( ~add_queue[10*4+0] &&
	     ( cmd_block_table1[ (add_queue[10*4+1:10*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[10*4+1:10*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[10*4+1:10*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[10*4+1:10*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[10*4+1:10*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[10*4+1:10*4+3])*62+61] )  ||
	   ( add_queue[10*4+0] &&
	     ( cmd_block_table2[ (add_queue[10*4+1:10*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[10*4+1:10*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[10*4+1:10*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[10*4+1:10*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[10*4+1:10*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[10*4+1:10*4+3])*62+61] )) ? 5'b1010 :
	   
	  (( ~add_queue[11*4+0] &&
	     ( cmd_block_table1[ (add_queue[11*4+1:11*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[11*4+1:11*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[11*4+1:11*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[11*4+1:11*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[11*4+1:11*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[11*4+1:11*4+3])*62+61] )  ||
	   ( add_queue[11*4+0] &&
	     ( cmd_block_table2[ (add_queue[11*4+1:11*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[11*4+1:11*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[11*4+1:11*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[11*4+1:11*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[11*4+1:11*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[11*4+1:11*4+3])*62+61] )) ? 5'b1011 :

	   (( ~add_queue[12*4+0] &&
	     ( cmd_block_table1[ (add_queue[12*4+1:12*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[12*4+1:12*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[12*4+1:12*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[12*4+1:12*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[12*4+1:12*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[12*4+1:12*4+3])*62+61] )  ||
	   ( add_queue[12*4+0] &&
	     ( cmd_block_table2[ (add_queue[12*4+1:12*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[12*4+1:12*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[12*4+1:12*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[12*4+1:12*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[12*4+1:12*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[12*4+1:12*4+3])*62+61] )) ? 5'b1100 :

	  (( ~add_queue[13*4+0] &&
	     ( cmd_block_table1[ (add_queue[13*4+1:13*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[13*4+1:13*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[13*4+1:13*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[13*4+1:13*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[13*4+1:13*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[13*4+1:13*4+3])*62+61] )  ||
	   ( add_queue[13*4+0] &&
	     ( cmd_block_table2[ (add_queue[13*4+1:13*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[13*4+1:13*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[13*4+1:13*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[13*4+1:13*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[13*4+1:13*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[13*4+1:13*4+3])*62+61] )) ? 5'b1101 :

	  (( ~add_queue[14*4+0] &&
	     ( cmd_block_table1[ (add_queue[14*4+1:14*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[14*4+1:14*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[14*4+1:14*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[14*4+1:14*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[14*4+1:14*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[14*4+1:14*4+3])*62+61] )  ||
	   ( add_queue[14*4+0] &&
	     ( cmd_block_table2[ (add_queue[14*4+1:14*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[14*4+1:14*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[14*4+1:14*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[14*4+1:14*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[14*4+1:14*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[14*4+1:14*4+3])*62+61] )) ? 5'b1110 :

	  (( ~add_queue[15*4+0] &&
	     ( cmd_block_table1[ (add_queue[15*4+1:15*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[15*4+1:15*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[15*4+1:15*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[15*4+1:15*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[15*4+1:15*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[15*4+1:15*4+3])*62+61] )  ||
	   ( add_queue[15*4+0] &&
	     ( cmd_block_table2[ (add_queue[15*4+1:15*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[15*4+1:15*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[15*4+1:15*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[15*4+1:15*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[15*4+1:15*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[15*4+1:15*4+3])*62+61] )) ? 5'b1111 :

	  (( ~add_queue[16*4+0] &&
	     ( cmd_block_table1[ (add_queue[16*4+1:16*4+3])*62+0] &&
	       ~cmd_block_table1[ (add_queue[16*4+1:16*4+3])*62+1] &&
	       ~cmd_block_table1[ (add_queue[16*4+1:16*4+3])*62+2] &&
	       ~cmd_block_table1[ (add_queue[16*4+1:16*4+3])*62+3] &&
	       ~cmd_block_table1[ (add_queue[16*4+1:16*4+3])*62+4] ) &&
	     ~cmd_block_table1[ (add_queue[16*4+1:16*4+3])*62+61] )  ||
	   ( add_queue[16*4+0] &&
	     ( cmd_block_table2[ (add_queue[16*4+1:16*4+3])*62+0] &&
	       ~cmd_block_table2[ (add_queue[16*4+1:16*4+3])*62+1] &&
	       ~cmd_block_table2[ (add_queue[16*4+1:16*4+3])*62+2] &&
	       ~cmd_block_table2[ (add_queue[16*4+1:16*4+3])*62+3] &&
	       ~cmd_block_table2[ (add_queue[16*4+1:16*4+3])*62+4] ) &&
	     ~cmd_block_table2[ (add_queue[16*4+1:16*4+3])*62+61] )) ? 5'b10000 :

	  5'b0;
   
//	same as temp_dispatch_add, but for shift cmds in the shift_queue
   assign temp_dispatch_shift = (reset) ? 5'b0 :
	  
	  (( ~shift_queue[1*4+0] &&
	     ( cmd_block_table1[ (shift_queue[1*4+1:1*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[1*4+1:1*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[1*4+1:1*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[1*4+1:1*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[1*4+1:1*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[1*4+1:1*4+3])*62+61] )  ||
	   ( shift_queue[1*4+0] &&
	     ( cmd_block_table2[ (shift_queue[1*4+1:1*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[1*4+1:1*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[1*4+1:1*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[1*4+1:1*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[1*4+1:1*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[1*4+1:1*4+3])*62+61] )) ? 5'b1 :

	  (( ~shift_queue[2*4+0] &&
	     ( cmd_block_table1[ (shift_queue[2*4+1:2*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[2*4+1:2*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[2*4+1:2*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[2*4+1:2*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[2*4+1:2*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[2*4+1:2*4+3])*62+61] )  ||
	   ( shift_queue[2*4+0] &&
	     ( cmd_block_table2[ (shift_queue[2*4+1:2*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[2*4+1:2*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[2*4+1:2*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[2*4+1:2*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[2*4+1:2*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[2*4+1:2*4+3])*62+61] )) ? 5'b10 :

	  (( ~shift_queue[3*4+0] &&
	     ( cmd_block_table1[ (shift_queue[3*4+1:3*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[3*4+1:3*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[3*4+1:3*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[3*4+1:3*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[3*4+1:3*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[3*4+1:3*4+3])*62+61] )  ||
	   ( shift_queue[3*4+0] &&
	     ( cmd_block_table2[ (shift_queue[3*4+1:3*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[3*4+1:3*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[3*4+1:3*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[3*4+1:3*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[3*4+1:3*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[3*4+1:3*4+3])*62+61] )) ? 5'b11 :
	   
	  (( ~shift_queue[4*4+0] &&
	     ( cmd_block_table1[ (shift_queue[4*4+1:4*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[4*4+1:4*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[4*4+1:4*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[4*4+1:4*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[4*4+1:4*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[4*4+1:4*4+3])*62+61] )  ||
	   ( shift_queue[4*4+0] &&
	     ( cmd_block_table2[ (shift_queue[4*4+1:4*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[4*4+1:4*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[4*4+1:4*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[4*4+1:4*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[4*4+1:4*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[4*4+1:4*4+3])*62+61] )) ? 5'b100 :

	   (( ~shift_queue[5*4+0] &&
	     ( cmd_block_table1[ (shift_queue[5*4+1:5*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[5*4+1:5*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[5*4+1:5*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[5*4+1:5*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[5*4+1:5*4+3])*62+4] ) &&
	      cmd_block_table1[ (shift_queue[5*4+1:5*4+3])*62+61] )  ||
	   ( shift_queue[5*4+0] &&
	     ( cmd_block_table2[ (shift_queue[5*4+1:5*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[5*4+1:5*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[5*4+1:5*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[5*4+1:5*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[5*4+1:5*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[5*4+1:5*4+3])*62+61] )) ? 5'b101 :

	  (( ~shift_queue[6*4+0] &&
	     ( cmd_block_table1[ (shift_queue[6*4+1:6*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[6*4+1:6*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[6*4+1:6*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[6*4+1:6*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[6*4+1:6*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[6*4+1:6*4+3])*62+61] )  ||
	   ( shift_queue[6*4+0] &&
	     ( cmd_block_table2[ (shift_queue[6*4+1:6*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[6*4+1:6*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[6*4+1:6*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[6*4+1:6*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[6*4+1:6*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[6*4+1:6*4+3])*62+61] )) ? 5'b110 :

	  (( ~shift_queue[7*4+0] &&
	     ( cmd_block_table1[ (shift_queue[7*4+1:7*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[7*4+1:7*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[7*4+1:7*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[7*4+1:7*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[7*4+1:7*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[7*4+1:7*4+3])*62+61] )  ||
	   ( shift_queue[7*4+0] &&
	     ( cmd_block_table2[ (shift_queue[7*4+1:7*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[7*4+1:7*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[7*4+1:7*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[7*4+1:7*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[7*4+1:7*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[7*4+1:7*4+3])*62+61] )) ? 5'b111 :

	   (( ~shift_queue[8*4+0] &&
	     ( cmd_block_table1[ (shift_queue[8*4+1:8*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[8*4+1:8*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[8*4+1:8*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[8*4+1:8*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[8*4+1:8*4+3])*62+4] ) &&
	      cmd_block_table1[ (shift_queue[8*4+1:8*4+3])*62+61] )  ||
	   ( shift_queue[8*4+0] &&
	     ( cmd_block_table2[ (shift_queue[8*4+1:8*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[8*4+1:8*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[8*4+1:8*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[8*4+1:8*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[8*4+1:8*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[8*4+1:8*4+3])*62+61] )) ? 5'b1000 :

	  (( ~shift_queue[9*4+0] &&
	     ( cmd_block_table1[ (shift_queue[9*4+1:9*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[9*4+1:9*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[9*4+1:9*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[9*4+1:9*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[9*4+1:9*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[9*4+1:9*4+3])*62+61] )  ||
	   ( shift_queue[9*4+0] &&
	     ( cmd_block_table2[ (shift_queue[9*4+1:9*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[9*4+1:9*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[9*4+1:9*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[9*4+1:9*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[9*4+1:9*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[9*4+1:9*4+3])*62+61] )) ? 5'b1001 :

	  (( ~shift_queue[10*4+0] &&
	     ( cmd_block_table1[ (shift_queue[10*4+1:10*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[10*4+1:10*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[10*4+1:10*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[10*4+1:10*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[10*4+1:10*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[10*4+1:10*4+3])*62+61] )  ||
	   ( shift_queue[10*4+0] &&
	     ( cmd_block_table2[ (shift_queue[10*4+1:10*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[10*4+1:10*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[10*4+1:10*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[10*4+1:10*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[10*4+1:10*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[10*4+1:10*4+3])*62+61] )) ? 5'b1010 :
	   
	  (( ~shift_queue[11*4+0] &&
	     ( cmd_block_table1[ (shift_queue[11*4+1:11*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[11*4+1:11*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[11*4+1:11*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[11*4+1:11*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[11*4+1:11*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[11*4+1:11*4+3])*62+61] )  ||
	   ( shift_queue[11*4+0] &&
	     ( cmd_block_table2[ (shift_queue[11*4+1:11*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[11*4+1:11*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[11*4+1:11*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[11*4+1:11*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[11*4+1:11*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[11*4+1:11*4+3])*62+61] )) ? 5'b1011 :

	   (( ~shift_queue[12*4+0] &&
	     ( cmd_block_table1[ (shift_queue[12*4+1:12*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[12*4+1:12*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[12*4+1:12*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[12*4+1:12*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[12*4+1:12*4+3])*62+4] ) &&
	      cmd_block_table1[ (shift_queue[12*4+1:12*4+3])*62+61] )  ||
	   ( shift_queue[12*4+0] &&
	     ( cmd_block_table2[ (shift_queue[12*4+1:12*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[12*4+1:12*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[12*4+1:12*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[12*4+1:12*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[12*4+1:12*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[12*4+1:12*4+3])*62+61] )) ? 5'b1100 :

	  (( ~shift_queue[13*4+0] &&
	     ( cmd_block_table1[ (shift_queue[13*4+1:13*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[13*4+1:13*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[13*4+1:13*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[13*4+1:13*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[13*4+1:13*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[13*4+1:13*4+3])*62+61] )  ||
	   ( shift_queue[13*4+0] &&
	     ( cmd_block_table2[ (shift_queue[13*4+1:13*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[13*4+1:13*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[13*4+1:13*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[13*4+1:13*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[13*4+1:13*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[13*4+1:13*4+3])*62+61] )) ? 5'b1101 :

	  (( ~shift_queue[14*4+0] &&
	     ( cmd_block_table1[ (shift_queue[14*4+1:14*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[14*4+1:14*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[14*4+1:14*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[14*4+1:14*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[14*4+1:14*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[14*4+1:14*4+3])*62+61] )  ||
	   ( shift_queue[14*4+0] &&
	     ( cmd_block_table2[ (shift_queue[14*4+1:14*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[14*4+1:14*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[14*4+1:14*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[14*4+1:14*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[14*4+1:14*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[14*4+1:14*4+3])*62+61] )) ? 5'b1110 :

	  (( ~shift_queue[15*4+0] &&
	     ( cmd_block_table1[ (shift_queue[15*4+1:15*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[15*4+1:15*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[15*4+1:15*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[15*4+1:15*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[15*4+1:15*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[15*4+1:15*4+3])*62+61] )  ||
	   ( shift_queue[15*4+0] &&
	     ( cmd_block_table2[ (shift_queue[15*4+1:15*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[15*4+1:15*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[15*4+1:15*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[15*4+1:15*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[15*4+1:15*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[15*4+1:15*4+3])*62+61] )) ? 5'b1111 :

	  (( ~shift_queue[16*4+0] &&
	     ( cmd_block_table1[ (shift_queue[16*4+1:16*4+3])*62+0] &&
	       ~cmd_block_table1[ (shift_queue[16*4+1:16*4+3])*62+1] &&
	       ~cmd_block_table1[ (shift_queue[16*4+1:16*4+3])*62+2] &&
	       ~cmd_block_table1[ (shift_queue[16*4+1:16*4+3])*62+3] &&
	       ~cmd_block_table1[ (shift_queue[16*4+1:16*4+3])*62+4] ) &&
	     cmd_block_table1[ (shift_queue[16*4+1:16*4+3])*62+61] )  ||
	   ( shift_queue[16*4+0] &&
	     ( cmd_block_table2[ (shift_queue[16*4+1:16*4+3])*62+0] &&
	       ~cmd_block_table2[ (shift_queue[16*4+1:16*4+3])*62+1] &&
	       ~cmd_block_table2[ (shift_queue[16*4+1:16*4+3])*62+2] &&
	       ~cmd_block_table2[ (shift_queue[16*4+1:16*4+3])*62+3] &&
	       ~cmd_block_table2[ (shift_queue[16*4+1:16*4+3])*62+4] ) &&
	     cmd_block_table2[ (shift_queue[16*4+1:16*4+3])*62+61] )) ? 5'b10000 :

	  5'b0;
   
	  

	  
// temp storage for add/shift conflict condition, used to determine if hazard will occur
   assign temp_hazard = ((add_queue[temp_dispatch_add*4+0] == add_queue[temp_dispatch_add*4+0]) &&
			 (add_queue[temp_dispatch_add*4+1] == add_queue[temp_dispatch_add*4+1]) &&
			 (add_queue[temp_dispatch_add*4+2] == add_queue[temp_dispatch_add*4+2]) &&
			 (add_queue[temp_dispatch_add*4+3] == add_queue[temp_dispatch_add*4+3])) ? 1'b1 :
	  1'b0;

   //set to 1 if the add and shift cmds to be dispatched will write to the same register

  /* always
     @ (temp_dispatch_add or temp_dispatch_shift or temp_hazard) begin
	case ({add_queue[temp_dispatch_add*4+1],add_queue[temp_dispatch_add*4+2],add_queue[temp_dispatch_add*4+3]})
	  'b000:
	    hazard1 = (reset) ? 1'b0 :
		      ((temp_dispatch_add != 5'b0) && (temp_dispatch_shift != 5'b0) && ~temp_hazard &&
		       
		       (( ~add_queue[temp_dispatch_add*4 + 0] &&
			 ~shift_queue[temp_dispatch_shift*4 + 0] &&
			 ((cmd_block_table1[0*62+24] == cmd_block_table1[0*62+24]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table1[0*62+25]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table1[0*62+26]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table1[0*62+27]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table1[0*62+28]))) ||

			( ~add_queue[temp_dispatch_add*4 + 0] &&
			 shift_queue[temp_dispatch_shift*4 + 0] &&
			 ((cmd_block_table1[0*62+24] == cmd_block_table2[0*62+24]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table2[0*62+25]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table2[0*62+26]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table2[0*62+27]) &&
			  (cmd_block_table1[0*62+24] == cmd_block_table2[0*62+28]))) ||

			( add_queue[temp_dispatch_add*4 + 0] &&
			 ~shift_queue[temp_dispatch_shift*4 + 0] &&
			 ((cmd_block_table2[0*62+24] == cmd_block_table1[0*62+24]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table1[0*62+25]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table1[0*62+26]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table1[0*62+27]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table1[0*62+28]))) ||

			( add_queue[temp_dispatch_add*4 + 0] &&
			 shift_queue[temp_dispatch_shift*4 + 0] &&
			 ((cmd_block_table2[0*62+24] == cmd_block_table2[0*62+24]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table2[0*62+25]) &&
  			  (cmd_block_table2[0*62+24] == cmd_block_table2[0*62+26]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table2[0*62+27]) &&
			  (cmd_block_table2[0*62+24] == cmd_block_table2[0*62+28]))))

	*/		


		       
   assign a[0] = add_queue[temp_dispatch_add*4 + 1];
   assign a[1] = add_queue[temp_dispatch_add*4 + 2];
   assign a[2] = add_queue[temp_dispatch_add*4 + 3];

   assign b[0] = shift_queue[temp_dispatch_shift*4+1];
   assign b[1] = shift_queue[temp_dispatch_shift*4+2];
   assign b[2] = shift_queue[temp_dispatch_shift*4+3];

   assign c1a[0:4] = { 5{1'b0}};
   assign c1a[5] = cmd_block_table1[ a[0:2]*62 + 5 ];
   assign c1a[6] = cmd_block_table1[ a[0:2]*62 + 6 ];
   assign c1a[7] = cmd_block_table1[ a[0:2]*62 + 7 ];
   assign c1a[8] = cmd_block_table1[ a[0:2]*62 + 8 ];
   assign c1a[9] = cmd_block_table1[ a[0:2]*62 + 9 ];
   assign c1a[10] = cmd_block_table1[ a[0:2]*62 + 10 ];
   assign c1a[11] = cmd_block_table1[ a[0:2]*62 + 11 ];
   assign c1a[12] = cmd_block_table1[ a[0:2]*62 + 12 ];
   assign c1a[13] = cmd_block_table1[ a[0:2]*62 + 13 ];
   assign c1a[14] = cmd_block_table1[ a[0:2]*62 + 14 ];
   assign c1a[15] = cmd_block_table1[ a[0:2]*62 + 15 ];
   assign c1a[16] = cmd_block_table1[ a[0:2]*62 + 16 ];
   assign c1a[17] = cmd_block_table1[ a[0:2]*62 + 17 ];
   assign c1a[18] = cmd_block_table1[ a[0:2]*62 + 18 ];
   assign c1a[19] = cmd_block_table1[ a[0:2]*62 + 19 ];
   assign c1a[20] = cmd_block_table1[ a[0:2]*62 + 20 ];
   assign c1a[20] = cmd_block_table1[ a[0:2]*62 + 20 ];
   assign c1a[21] = cmd_block_table1[ a[0:2]*62 + 21 ];
   assign c1a[22] = cmd_block_table1[ a[0:2]*62 + 22 ];
   assign c1a[23] = cmd_block_table1[ a[0:2]*62 + 23 ];
   assign c1a[24] = cmd_block_table1[ a[0:2]*62 + 24 ];
   assign c1a[25] = cmd_block_table1[ a[0:2]*62 + 25 ];
   assign c1a[26] = cmd_block_table1[ a[0:2]*62 + 26 ];
   assign c1a[27] = cmd_block_table1[ a[0:2]*62 + 27 ];
   assign c1a[28] = cmd_block_table1[ a[0:2]*62 + 28 ];
   assign c1a[29] = cmd_block_table1[ a[0:2]*62 + 29 ];
   assign c1a[30] = cmd_block_table1[ a[0:2]*62 + 30 ];
   assign c1a[31] = cmd_block_table1[ a[0:2]*62 + 31 ];
   assign c1a[32] = cmd_block_table1[ a[0:2]*62 + 32 ];
   assign c1a[33] = cmd_block_table1[ a[0:2]*62 + 33 ];
   assign c1a[34] = cmd_block_table1[ a[0:2]*62 + 34 ];
   assign c1a[35] = cmd_block_table1[ a[0:2]*62 + 35 ];
   assign c1a[36] = cmd_block_table1[ a[0:2]*62 + 36 ];
   assign c1a[37] = cmd_block_table1[ a[0:2]*62 + 37 ];
   assign c1a[38] = cmd_block_table1[ a[0:2]*62 + 38 ];
   assign c1a[39] = cmd_block_table1[ a[0:2]*62 + 39 ];
   assign c1a[40] = cmd_block_table1[ a[0:2]*62 + 40 ];
   assign c1a[41] = cmd_block_table1[ a[0:2]*62 + 41 ];
   assign c1a[42] = cmd_block_table1[ a[0:2]*62 + 42 ];
   assign c1a[43] = cmd_block_table1[ a[0:2]*62 + 43 ];
   assign c1a[44] = cmd_block_table1[ a[0:2]*62 + 44 ];
   assign c1a[45] = cmd_block_table1[ a[0:2]*62 + 45 ];
   assign c1a[46] = cmd_block_table1[ a[0:2]*62 + 46 ];
   assign c1a[47] = cmd_block_table1[ a[0:2]*62 + 47 ];
   assign c1a[48] = cmd_block_table1[ a[0:2]*62 + 48 ];
   assign c1a[49] = cmd_block_table1[ a[0:2]*62 + 49 ];
   assign c1a[50] = cmd_block_table1[ a[0:2]*62 + 50 ];
   assign c1a[51] = cmd_block_table1[ a[0:2]*62 + 51 ];
   assign c1a[52] = cmd_block_table1[ a[0:2]*62 + 52 ];
   assign c1a[53] = cmd_block_table1[ a[0:2]*62 + 53 ];
   assign c1a[54] = cmd_block_table1[ a[0:2]*62 + 54 ];
   assign c1a[55] = cmd_block_table1[ a[0:2]*62 + 55 ];
   assign c1a[56] = cmd_block_table1[ a[0:2]*62 + 56 ];
   assign c1a[57] = cmd_block_table1[ a[0:2]*62 + 57 ];
   assign c1a[58] = cmd_block_table1[ a[0:2]*62 + 58 ];
   assign c1a[59] = cmd_block_table1[ a[0:2]*62 + 59 ];
   assign c1a[60] = cmd_block_table1[ a[0:2]*62 + 60 ];
   assign c1a[61] = cmd_block_table1[ a[0:2]*62 + 61 ];

   assign c2a[0:4] = { 5{1'b0}};
   assign c2a[5] = cmd_block_table2[ a[0:2]*62 + 5 ];
   assign c2a[6] = cmd_block_table2[ a[0:2]*62 + 6 ];
   assign c2a[7] = cmd_block_table2[ a[0:2]*62 + 7 ];
   assign c2a[8] = cmd_block_table2[ a[0:2]*62 + 8 ];
   assign c2a[9] = cmd_block_table2[ a[0:2]*62 + 9 ];
   assign c2a[10] = cmd_block_table2[ a[0:2]*62 + 10 ];
   assign c2a[11] = cmd_block_table2[ a[0:2]*62 + 11 ];
   assign c2a[12] = cmd_block_table2[ a[0:2]*62 + 12 ];
   assign c2a[13] = cmd_block_table2[ a[0:2]*62 + 13 ];
   assign c2a[14] = cmd_block_table2[ a[0:2]*62 + 14 ];
   assign c2a[15] = cmd_block_table2[ a[0:2]*62 + 15 ];
   assign c2a[16] = cmd_block_table2[ a[0:2]*62 + 16 ];
   assign c2a[17] = cmd_block_table2[ a[0:2]*62 + 17 ];
   assign c2a[18] = cmd_block_table2[ a[0:2]*62 + 18 ];
   assign c2a[19] = cmd_block_table2[ a[0:2]*62 + 19 ];
   assign c2a[20] = cmd_block_table2[ a[0:2]*62 + 20 ];
   assign c2a[21] = cmd_block_table2[ a[0:2]*62 + 21 ];
   assign c2a[22] = cmd_block_table2[ a[0:2]*62 + 22 ];
   assign c2a[23] = cmd_block_table2[ a[0:2]*62 + 23 ];
   assign c2a[24] = cmd_block_table2[ a[0:2]*62 + 24 ];
   assign c2a[25] = cmd_block_table2[ a[0:2]*62 + 25 ];
   assign c2a[26] = cmd_block_table2[ a[0:2]*62 + 26 ];
   assign c2a[27] = cmd_block_table2[ a[0:2]*62 + 27 ];
   assign c2a[28] = cmd_block_table2[ a[0:2]*62 + 28 ];
   assign c2a[29] = cmd_block_table2[ a[0:2]*62 + 29 ];
   assign c2a[30] = cmd_block_table2[ a[0:2]*62 + 30 ];
   assign c2a[31] = cmd_block_table2[ a[0:2]*62 + 31 ];
   assign c2a[32] = cmd_block_table2[ a[0:2]*62 + 32 ];
   assign c2a[33] = cmd_block_table2[ a[0:2]*62 + 33 ];
   assign c2a[34] = cmd_block_table2[ a[0:2]*62 + 34 ];
   assign c2a[35] = cmd_block_table2[ a[0:2]*62 + 35 ];
   assign c2a[36] = cmd_block_table2[ a[0:2]*62 + 36 ];
   assign c2a[37] = cmd_block_table2[ a[0:2]*62 + 37 ];
   assign c2a[38] = cmd_block_table2[ a[0:2]*62 + 38 ];
   assign c2a[39] = cmd_block_table2[ a[0:2]*62 + 39 ];
   assign c2a[40] = cmd_block_table2[ a[0:2]*62 + 40 ];
   assign c2a[41] = cmd_block_table2[ a[0:2]*62 + 41 ];
   assign c2a[42] = cmd_block_table2[ a[0:2]*62 + 42 ];
   assign c2a[43] = cmd_block_table2[ a[0:2]*62 + 43 ];
   assign c2a[44] = cmd_block_table2[ a[0:2]*62 + 44 ];
   assign c2a[45] = cmd_block_table2[ a[0:2]*62 + 45 ];
   assign c2a[46] = cmd_block_table2[ a[0:2]*62 + 46 ];
   assign c2a[47] = cmd_block_table2[ a[0:2]*62 + 47 ];
   assign c2a[48] = cmd_block_table2[ a[0:2]*62 + 48 ];
   assign c2a[49] = cmd_block_table2[ a[0:2]*62 + 49 ];
   assign c2a[50] = cmd_block_table2[ a[0:2]*62 + 50 ];
   assign c2a[51] = cmd_block_table2[ a[0:2]*62 + 51 ];
   assign c2a[52] = cmd_block_table2[ a[0:2]*62 + 52 ];
   assign c2a[53] = cmd_block_table2[ a[0:2]*62 + 53 ];
   assign c2a[54] = cmd_block_table2[ a[0:2]*62 + 54 ];
   assign c2a[55] = cmd_block_table2[ a[0:2]*62 + 55 ];
   assign c2a[56] = cmd_block_table2[ a[0:2]*62 + 56 ];
   assign c2a[57] = cmd_block_table2[ a[0:2]*62 + 57 ];
   assign c2a[58] = cmd_block_table2[ a[0:2]*62 + 58 ];
   assign c2a[59] = cmd_block_table2[ a[0:2]*62 + 59 ];
   assign c2a[60] = cmd_block_table2[ a[0:2]*62 + 60 ];
   assign c2a[61] = cmd_block_table2[ a[0:2]*62 + 61 ];

   assign c1b[0:4] = { 5{1'b0}};
   assign c1b[5] = cmd_block_table1[ b[0:2]*62 + 5 ];
   assign c1b[6] = cmd_block_table1[ b[0:2]*62 + 6 ];
   assign c1b[7] = cmd_block_table1[ b[0:2]*62 + 7 ];
   assign c1b[8] = cmd_block_table1[ b[0:2]*62 + 8 ];
   assign c1b[9] = cmd_block_table1[ b[0:2]*62 + 9 ];
   assign c1b[10] = cmd_block_table1[ b[0:2]*62 + 10 ];
   assign c1b[11] = cmd_block_table1[ b[0:2]*62 + 11 ];
   assign c1b[12] = cmd_block_table1[ b[0:2]*62 + 12 ];
   assign c1b[13] = cmd_block_table1[ b[0:2]*62 + 13 ];
   assign c1b[14] = cmd_block_table1[ b[0:2]*62 + 14 ];
   assign c1b[15] = cmd_block_table1[ b[0:2]*62 + 15 ];
   assign c1b[16] = cmd_block_table1[ b[0:2]*62 + 16 ];
   assign c1b[17] = cmd_block_table1[ b[0:2]*62 + 17 ];
   assign c1b[18] = cmd_block_table1[ b[0:2]*62 + 18 ];
   assign c1b[19] = cmd_block_table1[ b[0:2]*62 + 19 ];
   assign c1b[20] = cmd_block_table1[ b[0:2]*62 + 20 ];
   assign c1b[21] = cmd_block_table1[ b[0:2]*62 + 21 ];
   assign c1b[22] = cmd_block_table1[ b[0:2]*62 + 22 ];
   assign c1b[23] = cmd_block_table1[ b[0:2]*62 + 23 ];
   assign c1b[24] = cmd_block_table1[ b[0:2]*62 + 24 ];
   assign c1b[25] = cmd_block_table1[ b[0:2]*62 + 25 ];
   assign c1b[26] = cmd_block_table1[ b[0:2]*62 + 26 ];
   assign c1b[27] = cmd_block_table1[ b[0:2]*62 + 27 ];
   assign c1b[28] = cmd_block_table1[ b[0:2]*62 + 28 ];
   assign c1b[29] = cmd_block_table1[ b[0:2]*62 + 29 ];
   assign c1b[30] = cmd_block_table1[ b[0:2]*62 + 30 ];
   assign c1b[31] = cmd_block_table1[ b[0:2]*62 + 31 ];
   assign c1b[32] = cmd_block_table1[ b[0:2]*62 + 32 ];
   assign c1b[33] = cmd_block_table1[ b[0:2]*62 + 33 ];
   assign c1b[34] = cmd_block_table1[ b[0:2]*62 + 34 ];
   assign c1b[35] = cmd_block_table1[ b[0:2]*62 + 35 ];
   assign c1b[36] = cmd_block_table1[ b[0:2]*62 + 36 ];
   assign c1b[37] = cmd_block_table1[ b[0:2]*62 + 37 ];
   assign c1b[38] = cmd_block_table1[ b[0:2]*62 + 38 ];
   assign c1b[39] = cmd_block_table1[ b[0:2]*62 + 39 ];
   assign c1b[40] = cmd_block_table1[ b[0:2]*62 + 40 ];
   assign c1b[41] = cmd_block_table1[ b[0:2]*62 + 41 ];
   assign c1b[42] = cmd_block_table1[ b[0:2]*62 + 42 ];
   assign c1b[43] = cmd_block_table1[ b[0:2]*62 + 43 ];
   assign c1b[44] = cmd_block_table1[ b[0:2]*62 + 44 ];
   assign c1b[45] = cmd_block_table1[ b[0:2]*62 + 45 ];
   assign c1b[46] = cmd_block_table1[ b[0:2]*62 + 46 ];
   assign c1b[47] = cmd_block_table1[ b[0:2]*62 + 47 ];
   assign c1b[48] = cmd_block_table1[ b[0:2]*62 + 48 ];
   assign c1b[49] = cmd_block_table1[ b[0:2]*62 + 49 ];
   assign c1b[50] = cmd_block_table1[ b[0:2]*62 + 50 ];
   assign c1b[51] = cmd_block_table1[ b[0:2]*62 + 51 ];
   assign c1b[52] = cmd_block_table1[ b[0:2]*62 + 52 ];
   assign c1b[53] = cmd_block_table1[ b[0:2]*62 + 53 ];
   assign c1b[54] = cmd_block_table1[ b[0:2]*62 + 54 ];
   assign c1b[55] = cmd_block_table1[ b[0:2]*62 + 55 ];
   assign c1b[56] = cmd_block_table1[ b[0:2]*62 + 56 ];
   assign c1b[57] = cmd_block_table1[ b[0:2]*62 + 57 ];
   assign c1b[58] = cmd_block_table1[ b[0:2]*62 + 58 ];
   assign c1b[59] = cmd_block_table1[ b[0:2]*62 + 59 ];
   assign c1b[60] = cmd_block_table1[ b[0:2]*62 + 60 ];
   assign c1b[61] = cmd_block_table1[ b[0:2]*62 + 61 ];

   assign c2b[0:4] = { 5{1'b0}};
   assign c2b[5] = cmd_block_table2[ b[0:2]*62 + 5 ];
   assign c2b[6] = cmd_block_table2[ b[0:2]*62 + 6 ];
   assign c2b[7] = cmd_block_table2[ b[0:2]*62 + 7 ];
   assign c2b[8] = cmd_block_table2[ b[0:2]*62 + 8 ];
   assign c2b[9] = cmd_block_table2[ b[0:2]*62 + 9 ];
   assign c2b[10] = cmd_block_table2[ b[0:2]*62 + 10 ];
   assign c2b[11] = cmd_block_table2[ b[0:2]*62 + 11 ];
   assign c2b[12] = cmd_block_table2[ b[0:2]*62 + 12 ];
   assign c2b[13] = cmd_block_table2[ b[0:2]*62 + 13 ];
   assign c2b[14] = cmd_block_table2[ b[0:2]*62 + 14 ];
   assign c2b[15] = cmd_block_table2[ b[0:2]*62 + 15 ];
   assign c2b[16] = cmd_block_table2[ b[0:2]*62 + 16 ];
   assign c2b[17] = cmd_block_table2[ b[0:2]*62 + 17 ];
   assign c2b[18] = cmd_block_table2[ b[0:2]*62 + 18 ];
   assign c2b[19] = cmd_block_table2[ b[0:2]*62 + 19 ];
   assign c2b[20] = cmd_block_table2[ b[0:2]*62 + 20 ];
   assign c2b[21] = cmd_block_table2[ b[0:2]*62 + 21 ];
   assign c2b[22] = cmd_block_table2[ b[0:2]*62 + 22 ];
   assign c2b[23] = cmd_block_table2[ b[0:2]*62 + 23 ];
   assign c2b[24] = cmd_block_table2[ b[0:2]*62 + 24 ];
   assign c2b[25] = cmd_block_table2[ b[0:2]*62 + 25 ];
   assign c2b[26] = cmd_block_table2[ b[0:2]*62 + 26 ];
   assign c2b[27] = cmd_block_table2[ b[0:2]*62 + 27 ];
   assign c2b[28] = cmd_block_table2[ b[0:2]*62 + 28 ];
   assign c2b[29] = cmd_block_table2[ b[0:2]*62 + 29 ];
   assign c2b[30] = cmd_block_table2[ b[0:2]*62 + 30 ];
   assign c2b[31] = cmd_block_table2[ b[0:2]*62 + 31 ];
   assign c2b[32] = cmd_block_table2[ b[0:2]*62 + 32 ];
   assign c2b[33] = cmd_block_table2[ b[0:2]*62 + 33 ];
   assign c2b[34] = cmd_block_table2[ b[0:2]*62 + 34 ];
   assign c2b[35] = cmd_block_table2[ b[0:2]*62 + 35 ];
   assign c2b[36] = cmd_block_table2[ b[0:2]*62 + 36 ];
   assign c2b[37] = cmd_block_table2[ b[0:2]*62 + 37 ];
   assign c2b[38] = cmd_block_table2[ b[0:2]*62 + 38 ];
   assign c2b[39] = cmd_block_table2[ b[0:2]*62 + 39 ];
   assign c2b[40] = cmd_block_table2[ b[0:2]*62 + 40 ];
   assign c2b[41] = cmd_block_table2[ b[0:2]*62 + 41 ];
   assign c2b[42] = cmd_block_table2[ b[0:2]*62 + 42 ];
   assign c2b[43] = cmd_block_table2[ b[0:2]*62 + 43 ];
   assign c2b[44] = cmd_block_table2[ b[0:2]*62 + 44 ];
   assign c2b[45] = cmd_block_table2[ b[0:2]*62 + 45 ];
   assign c2b[46] = cmd_block_table2[ b[0:2]*62 + 46 ];
   assign c2b[47] = cmd_block_table2[ b[0:2]*62 + 47 ];
   assign c2b[48] = cmd_block_table2[ b[0:2]*62 + 48 ];
   assign c2b[49] = cmd_block_table2[ b[0:2]*62 + 49 ];
   assign c2b[50] = cmd_block_table2[ b[0:2]*62 + 50 ];
   assign c2b[51] = cmd_block_table2[ b[0:2]*62 + 51 ];
   assign c2b[52] = cmd_block_table2[ b[0:2]*62 + 52 ];
   assign c2b[53] = cmd_block_table2[ b[0:2]*62 + 53 ];
   assign c2b[54] = cmd_block_table2[ b[0:2]*62 + 54 ];
   assign c2b[55] = cmd_block_table2[ b[0:2]*62 + 55 ];
   assign c2b[56] = cmd_block_table2[ b[0:2]*62 + 56 ];
   assign c2b[57] = cmd_block_table2[ b[0:2]*62 + 57 ];
   assign c2b[58] = cmd_block_table2[ b[0:2]*62 + 58 ];
   assign c2b[59] = cmd_block_table2[ b[0:2]*62 + 59 ];
   assign c2b[60] = cmd_block_table2[ b[0:2]*62 + 60 ];
   assign c2b[61] = cmd_block_table2[ b[0:2]*62 + 61 ];


   //set to 1 if the add and shift cmds to be dispatched will write to the same register
   assign hazard1 = (reset) ? 1'b0 :
	  ((temp_dispatch_add != 5'b0) && (temp_dispatch_shift != 5'b0) && ~temp_hazard &&
	   
	   (( ~add_queue[temp_dispatch_add*4 + 0] &&
	      ~shift_queue[temp_dispatch_shift*4 + 0] &&
	      (c1a[24:28] == c1b[24:28])) ||

	   ( ~add_queue[temp_dispatch_add*4 + 0] &&
	     shift_queue[temp_dispatch_shift*4 + 0] &&
	     (c1a[24:28] == c2b[24:28])) ||

	    ( add_queue[temp_dispatch_add*4 + 0] &&
	      ~shift_queue[temp_dispatch_shift*4 + 0] &&
	      (c2a[24:28] == c1b[24:28])) ||

	    ( add_queue[temp_dispatch_add*4 + 0] &&
	      shift_queue[temp_dispatch_shift*4 + 0] &&
	      (c2a[24:28] == c2b[24:28])))) ? 1'b1 :
	  1'b0;
   

   // set to 1 if the add and shift cmds to be dispatched are from the same port
   assign hazard2 = (reset) ? 1'b0 :
	  ((temp_dispatch_add != 5'b0) && (temp_dispatch_shift != 5'b0) && 
	   ((add_queue[temp_dispatch_add*4+0] == shift_queue[temp_dispatch_shift*4+0]) &&
	    (add_queue[temp_dispatch_add*4+1] == shift_queue[temp_dispatch_shift*4+1]))) ? 1'b1 :
	  1'b0;
   
   // set to the index of the cmd in the add_queue that will be dispatched, avoiding the two hazard conditions above
   assign dispatch_add = (reset) ? 5'b0 :
	  ( add_or_shift && (hazard1 || hazard2)) ? 5'b0 :
	  temp_dispatch_add;


   // set to the index of the cmd in the shift_queue that will be dispatched, avoiding the two hazard conditions above
   assign dispatch_shift = (reset) ? 5'b0 :
	  ( ~add_or_shift && (hazard1 || hazard2)) ? 5'b0 :
	  temp_dispatch_shift;
//   assign dispatch_shift = 1'b1;
   


   // determine if there is a read/write conflict
   assign rd_wr_conflict = (( cmd_block_table1[ a[0:2]*62 + 24] != cmd_block_table1[ a[0:2]*62 + 24]) &&
			    ( cmd_block_table1[ a[0:2]*62 + 25] != cmd_block_table1[ a[0:2]*62 + 25]) &&
			    ( cmd_block_table1[ a[0:2]*62 + 26] != cmd_block_table1[ a[0:2]*62 + 26]) &&
			    ( cmd_block_table1[ a[0:2]*62 + 27] != cmd_block_table1[ a[0:2]*62 + 27]) &&
			    ( cmd_block_table1[ a[0:2]*62 + 28] != cmd_block_table1[ a[0:2]*62 + 28])) ? 1'b1 :
	  1'b0;
   

/*   assign rd_wr_conflict = (( cmd_block_table1[  (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+24] !=
		      cmd_block_table1[ (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+24]) &&
		     ( cmd_block_table1[  (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+25] !=
		      cmd_block_table1[ (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+25]) &&
		     ( cmd_block_table1[  (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+26] !=
		      cmd_block_table1[ (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+26]) &&
		     ( cmd_block_table1[  (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+27] !=
		      cmd_block_table1[ (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+27]) &&
		     ( cmd_block_table1[  (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+28] !=
		      cmd_block_table1[ (add_queue[dispatch_add*4+1:dispatch_add*4+3])*62+28])) ? 1'b1 :
                                     1'b0;
*/

   // special case where read/write conflict can occur  
   assign rw_case = ((cmd_block_table2[ b[0:2]*62+24 ] != cmd_block_table2[ b[0:2]*62 + 24 ]) &&
			     (cmd_block_table2[ b[0:2]*62+25 ] != cmd_block_table2[ b[0:2]*62 + 25 ]) &&
			     (cmd_block_table2[ b[0:2]*62+26 ] != cmd_block_table2[ b[0:2]*62 + 26 ]) &&
			     (cmd_block_table2[ b[0:2]*62+27 ] != cmd_block_table2[ b[0:2]*62 + 27 ]) &&
			     (cmd_block_table2[ b[0:2]*62+28 ] != cmd_block_table2[ b[0:2]*62 + 28 ])) ? 1'b1 :
		  1'b0;

   
   
   // store the register that the dispatched add cmd will write back to, so that a cmd that reads from that register will not
   // be dispatched on the next cycle
   assign rd_wr_conflict_block1 = (reset) ? 5'b0 :
	  ((dispatch_add != 5'b0) && ~add_queue[ dispatch_add*4 + 0 ] && rd_wr_conflict) ? c1a[24:28] :
	  ((dispatch_add != 5'b0) && add_queue[ dispatch_add*4 + 0 ] && rd_wr_conflict )  ?  c2a[24:28] :
	  5'b0;

   // store the register that the dispatched add cmd will write back to, so that a cmd that reads from that register will not
   // be dispatched on the next cycle
   assign rd_wr_conflict_block2 = (reset) ? 5'b0 :
	  ((dispatch_shift != 5'b0) && ~shift_queue[ dispatch_shift*4 + 0 ] && rd_wr_conflict) ? c1b[24:28] :
	  ((dispatch_shift != 5'b0) && shift_queue[ dispatch_shift*4 + 0 ] && rd_wr_conflict) ? c2b[24:28] :
	  5'b0;
   
//determine if there is an add and shift to be dispatched this cycle
   assign add_and_shift = ((temp_dispatch_add != 5'b0) &&
			   (temp_dispatch_shift != 5'b0)   &&  
			   (temp_dispatch_add == 5'b0) && 
			   (temp_dispatch_shift == 5'b0))  ? 1'b1:
	  1'b0;

   assign rw_case0 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold1_data1 }) && (hold1_tag == 2'b0) && (hold1_cmd != 'b1001) && rw_case &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b0) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold1_data1}) && (hold1_tag == 2'b0) && (hold1_cmd != 'b1001) && 
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b0) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   assign rw_case1 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold1_data1 }) && (hold1_tag == 2'b01) && (hold1_cmd != 'b1001) && rw_case &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b01) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold1_data1}) && (hold1_tag == 2'b01) && (hold1_cmd != 'b1001) && 
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b01) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
   
   assign rw_case2 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold1_data1 }) && (hold1_tag == 2'b10) && (hold1_cmd != 'b1001) && rw_case &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b10) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold1_data1}) && (hold1_tag == 2'b10) && (hold1_cmd != 'b1001) && 
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b10) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
		 
   assign rw_case3 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold1_data1 }) && (hold1_tag == 2'b11) && (hold1_cmd != 'b1001) && rw_case &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b11) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold1_data1}) && (hold1_tag == 2'b11) && (hold1_cmd != 'b1001) && 
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold1_data2}) && (hold1_tag == 2'b11) && (hold1_cmd != 'b1001) && (hold1_cmd != 'b1010) && (hold1_cmd != 'b1100) &&
	    ((hold1_add_valid == 5'b1) || (hold1_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   
   assign rw_case4 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold2_data1 }) && (hold2_tag == 2'b0) && (hold2_cmd != 'b1001) && rw_case &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b0) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold2_data1}) && (hold2_tag == 2'b0) && (hold2_cmd != 'b1001) && 
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b0) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   assign rw_case5 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold2_data1 }) && (hold2_tag == 2'b01) && (hold2_cmd != 'b1001) && rw_case &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b01) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold2_data1}) && (hold2_tag == 2'b01) && (hold2_cmd != 'b1001) && 
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b01) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
   
   assign rw_case6 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold2_data1 }) && (hold2_tag == 2'b10) && (hold2_cmd != 'b1001) && rw_case &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b10) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold2_data1}) && (hold2_tag == 2'b10) && (hold2_cmd != 'b1001) && 
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b10) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
		 
   assign rw_case7 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold2_data1 }) && (hold2_tag == 2'b11) && (hold2_cmd != 'b1001) && rw_case &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b11) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold2_data1}) && (hold2_tag == 2'b11) && (hold2_cmd != 'b1001) && 
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold2_data2}) && (hold2_tag == 2'b11) && (hold2_cmd != 'b1001) && (hold2_cmd != 'b1010) && (hold2_cmd != 'b1100) &&
	    ((hold2_add_valid == 5'b1) || (hold2_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   assign rw_case8 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold3_data1 }) && (hold3_tag == 2'b0) && (hold3_cmd != 'b1001) && rw_case &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b0) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold3_data1}) && (hold3_tag == 2'b0) && (hold3_cmd != 'b1001) && 
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b0) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   assign rw_case9 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold3_data1 }) && (hold3_tag == 2'b01) && (hold3_cmd != 'b1001) && rw_case &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b01) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold3_data1}) && (hold3_tag == 2'b01) && (hold3_cmd != 'b1001) && 
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b01) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
   
   assign rw_case10 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold3_data1 }) && (hold3_tag == 2'b10) && (hold3_cmd != 'b1001) && rw_case &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b10) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold3_data1}) && (hold3_tag == 2'b10) && (hold3_cmd != 'b1001) && 
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b10) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
		 
   assign rw_case11 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold3_data1 }) && (hold3_tag == 2'b11) && (hold3_cmd != 'b1001) && rw_case &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b11) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold3_data1}) && (hold3_tag == 2'b11) && (hold3_cmd != 'b1001) && 
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold3_data2}) && (hold3_tag == 2'b11) && (hold3_cmd != 'b1001) && (hold3_cmd != 'b1010) && (hold3_cmd != 'b1100) &&
	    ((hold3_add_valid == 5'b1) || (hold3_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   
   assign rw_case12 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold4_data1 }) && (hold4_tag == 2'b0) && (hold4_cmd != 'b1001) && rw_case &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b0) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold4_data1}) && (hold4_tag == 2'b0) && (hold4_cmd != 'b1001) && 
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b0) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   assign rw_case13 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold4_data1 }) && (hold4_tag == 2'b01) && (hold4_cmd != 'b1001) && rw_case &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b01) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold4_data1}) && (hold4_tag == 2'b01) && (hold4_cmd != 'b1001) && 
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b01) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
   
   assign rw_case14 = 
	  (((rd_wr_conflict_block1 == { 1'b1, hold4_data1 }) && (hold4_tag == 2'b10) && (hold4_cmd != 'b1001) && rw_case &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b10) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold4_data1}) && (hold4_tag == 2'b10) && (hold4_cmd != 'b1001) && 
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b10) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;
		 
   assign rw_case15 =
	   (((rd_wr_conflict_block1 == { 1'b1, hold4_data1 }) && (hold4_tag == 2'b11) && (hold4_cmd != 'b1001) && rw_case &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b11) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block2 == { 1'b1 ,  hold4_data1}) && (hold4_tag == 2'b11) && (hold4_cmd != 'b1001) && 
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1))) ||
	  ((rd_wr_conflict_block1 == { 1'b1 ,  hold4_data2}) && (hold4_tag == 2'b11) && (hold4_cmd != 'b1001) && (hold4_cmd != 'b1010) && (hold4_cmd != 'b1100) &&
	    ((hold4_add_valid == 5'b1) || (hold4_shift_valid == 5'b1)))) ? 1'b1 :
	  1'b0;

   always
     @ (negedge c_clk) begin

	// addqueue_curpos holds the number of valid commands in the addqueue at this time
	// This is where it's calculated for the next cycle.
	// A value of "00000" means there are no commands in the addqueue
	// A value of "10000" means it's filled, but that can't happen because of throughput

	addqueue_curpos <= (reset) ? 5'b0 :
			    ((addqueue_curpos == 5'b0) || (dispatch_add == 5'b0)) ?
			    addqueue_curpos + hold1_add_valid + hold2_add_valid + hold3_add_valid + hold4_add_valid :
			    addqueue_curpos + hold1_add_valid + hold2_add_valid + hold3_add_valid + hold4_add_valid -1;

	// shiftqueue_curpos holds the number of valid commands in the addqueue at this time
	// This is where it's calculated for the next cycle.
	// A value of "00000" means there are no commands in the addqueue
	// A value of "10000" means it's filled, but that can't happen because of throughput
   	shiftqueue_curpos <= (reset) ? 5'b0 :
			    ((shiftqueue_curpos == 5'b0) || (dispatch_shift == 5'b0)) ?
			    shiftqueue_curpos + hold1_shift_valid + hold2_shift_valid + hold3_shift_valid + hold4_shift_valid :
			    shiftqueue_curpos + hold1_shift_valid + hold2_shift_valid + hold3_shift_valid + hold4_shift_valid -1;

	//The case above take into account that 1 command will be dispatched (hence the '-1')

	//toggle between blocking adds and shifts due to hazards
	add_or_shift <= (reset) ? 1'b0 :
			( ~add_or_shift && (hazard1 | hazard2) && add_and_shift) ? 1'b1 :
			( add_or_shift && (hazard1 | hazard2) && add_and_shift) ? 1'b0 :
			add_or_shift;


	//If the new command from the holdreg is invalid, then set the holdx_invalid registers;
	hold1_invalid <=
			((hold1_add_valid == 5'b0) && 
			 (hold1_shift_valid == 5'b0) && 
			 (hold1_cmd != 4'b0)) ? {1'b1 , hold1_tag[0:1]} :
			3'b0;
	hold2_invalid <=
			((hold2_add_valid == 5'b0) && 
			 (hold2_shift_valid == 5'b0) && 
			 (hold2_cmd != 4'b0)) ? {1'b1 , hold2_tag[0:1]} :
			3'b0;
	hold3_invalid <=
			((hold3_add_valid == 5'b0) && 
			 (hold3_shift_valid == 5'b0) && 
			 (hold3_cmd != 4'b0)) ? {1'b1 , hold3_tag[0:1]} :
			3'b0;
	hold4_invalid <=
			((hold4_add_valid == 5'b0) && 
			 (hold4_shift_valid == 5'b0) && 
			 (hold4_cmd != 4'b0)) ? {1'b1 , hold4_tag[0:1]} :
			3'b0;
	
     end // always @ (negedge c_clk)

   // SET ALL OUTPUTS BELOW
   // Set the tag output to the mux_out for invalid opcodes
   assign port1_invalid_op = hold1_invalid[0];
   assign port1_invalid_tag = hold1_invalid[1:2];
   assign port2_invalid_op = hold2_invalid[0];
   assign port2_invalid_tag = hold2_invalid[1:2];
   assign port3_invalid_op = hold3_invalid[0];
   assign port3_invalid_tag = hold3_invalid[1:2];
   assign port4_invalid_op = hold4_invalid[0];
   assign port4_invalid_tag = hold4_invalid[1:2];

   assign add_dispatch_pointer[0] = add_queue[ dispatch_add*4 + 0 ];
   assign add_dispatch_pointer[1] = add_queue[ dispatch_add*4 + 1 ];
   assign add_dispatch_pointer[2] = add_queue[ dispatch_add*4 + 2 ];
   assign add_dispatch_pointer[3] = add_queue[ dispatch_add*4 + 3 ];

   assign shift_dispatch_pointer[0] = shift_queue[ dispatch_shift*4 + 0 ];
   assign shift_dispatch_pointer[1] = shift_queue[ dispatch_shift*4 + 1 ];
   assign shift_dispatch_pointer[2] = shift_queue[ dispatch_shift*4 + 2 ];
   assign shift_dispatch_pointer[3] = shift_queue[ dispatch_shift*4 + 3 ];

   assign c1a2[0:4] = 0;
   assign c1a2[5] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 5];
   assign c1a2[6] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 6];
   assign c1a2[7] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 7];
   assign c1a2[8] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 8];
   assign c1a2[9] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 9];
   assign c1a2[10] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 10];
   assign c1a2[11] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 11];
   assign c1a2[12] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 12];
   assign c1a2[13] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 13];
   assign c1a2[14] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 14];
   assign c1a2[15] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 15];
   assign c1a2[16] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 16];
   assign c1a2[17] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 17];
   assign c1a2[18] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 18];
   assign c1a2[19] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 19];
   assign c1a2[20] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 20];
   assign c1a2[21] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 21];
   assign c1a2[22] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 22];
   assign c1a2[23] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 23];
   assign c1a2[24] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 24];
   assign c1a2[25] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 25];
   assign c1a2[26] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 26];
   assign c1a2[27] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 27];
   assign c1a2[28] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 28];
   assign c1a2[29] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 29];
   assign c1a2[30] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 30];
   assign c1a2[31] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 31];
   assign c1a2[32] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 32];
   assign c1a2[33] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 33];
   assign c1a2[34] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 34];
   assign c1a2[35] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 35];
   assign c1a2[36] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 36];
   assign c1a2[37] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 37];
   assign c1a2[38] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 38];
   assign c1a2[39] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 39];
   assign c1a2[40] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 40];
   assign c1a2[41] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 41];
   assign c1a2[42] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 42];
   assign c1a2[43] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 43];
   assign c1a2[44] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 44];
   assign c1a2[45] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 45];
   assign c1a2[46] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 46];
   assign c1a2[47] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 47];
   assign c1a2[48] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 48];
   assign c1a2[49] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 49];
   assign c1a2[50] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 50];
   assign c1a2[51] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 51];
   assign c1a2[52] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 52];
   assign c1a2[53] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 53];
   assign c1a2[54] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 54];
   assign c1a2[55] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 55];
   assign c1a2[56] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 56];
   assign c1a2[57] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 57];
   assign c1a2[58] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 58];
   assign c1a2[59] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 59];
   assign c1a2[60] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 60];
   assign c1a2[61] = cmd_block_table1[add_dispatch_pointer[1:3]*62 + 61];

   assign c1b2[0:4] = 0;
   assign c1b2[5] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 5];
   assign c1b2[6] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 6];
   assign c1b2[7] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 7];
   assign c1b2[8] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 8];
   assign c1b2[9] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 9];
   assign c1b2[10] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 10];
   assign c1b2[11] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 11];
   assign c1b2[12] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 12];
   assign c1b2[13] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 13];
   assign c1b2[14] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 14];
   assign c1b2[15] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 15];
   assign c1b2[16] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 16];
   assign c1b2[17] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 17];
   assign c1b2[18] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 18];
   assign c1b2[19] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 19];
   assign c1b2[20] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 20];
   assign c1b2[21] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 21];
   assign c1b2[22] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 22];
   assign c1b2[23] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 23];
   assign c1b2[24] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 24];
   assign c1b2[25] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 25];
   assign c1b2[26] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 26];
   assign c1b2[27] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 27];
   assign c1b2[28] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 28];
   assign c1b2[29] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 29];
   assign c1b2[30] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 30];
   assign c1b2[31] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 31];
   assign c1b2[32] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 32];
   assign c1b2[33] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 33];
   assign c1b2[34] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 34];
   assign c1b2[35] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 35];
   assign c1b2[36] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 36];
   assign c1b2[37] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 37];
   assign c1b2[38] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 38];
   assign c1b2[39] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 39];
   assign c1b2[40] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 40];
   assign c1b2[41] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 41];
   assign c1b2[42] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 42];
   assign c1b2[43] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 43];
   assign c1b2[44] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 44];
   assign c1b2[45] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 45];
   assign c1b2[46] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 46];
   assign c1b2[47] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 47];
   assign c1b2[48] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 48];
   assign c1b2[49] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 49];
   assign c1b2[50] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 50];
   assign c1b2[51] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 51];
   assign c1b2[52] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 52];
   assign c1b2[53] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 53];
   assign c1b2[54] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 54];
   assign c1b2[55] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 55];
   assign c1b2[56] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 56];
   assign c1b2[57] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 57];
   assign c1b2[58] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 58];
   assign c1b2[59] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 59];
   assign c1b2[60] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 60];
   assign c1b2[61] = cmd_block_table1[shift_dispatch_pointer[1:3]*62 + 61];

   assign c2a2[0:4] = 0;
   assign c2a2[5] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 5];
   assign c2a2[6] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 6];
   assign c2a2[7] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 7];
   assign c2a2[8] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 8];
   assign c2a2[9] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 9];
   assign c2a2[10] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 10];
   assign c2a2[11] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 11];
   assign c2a2[12] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 12];
   assign c2a2[13] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 13];
   assign c2a2[14] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 14];
   assign c2a2[15] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 15];
   assign c2a2[16] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 16];
   assign c2a2[17] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 17];
   assign c2a2[18] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 18];
   assign c2a2[19] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 19];
   assign c2a2[20] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 20];
   assign c2a2[21] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 21];
   assign c2a2[22] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 22];
   assign c2a2[23] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 23];
   assign c2a2[24] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 24];
   assign c2a2[25] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 25];
   assign c2a2[26] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 26];
   assign c2a2[27] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 27];
   assign c2a2[28] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 28];
   assign c2a2[29] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 29];
   assign c2a2[30] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 30];
   assign c2a2[31] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 31];
   assign c2a2[32] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 32];
   assign c2a2[33] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 33];
   assign c2a2[34] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 34];
   assign c2a2[35] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 35];
   assign c2a2[36] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 36];
   assign c2a2[37] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 37];
   assign c2a2[38] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 38];
   assign c2a2[39] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 39];
   assign c2a2[40] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 40];
   assign c2a2[41] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 41];
   assign c2a2[42] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 42];
   assign c2a2[43] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 43];
   assign c2a2[44] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 44];
   assign c2a2[45] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 45];
   assign c2a2[46] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 46];
   assign c2a2[47] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 47];
   assign c2a2[48] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 48];
   assign c2a2[49] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 49];
   assign c2a2[50] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 50];
   assign c2a2[51] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 51];
   assign c2a2[52] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 52];
   assign c2a2[53] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 53];
   assign c2a2[54] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 54];
   assign c2a2[55] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 55];
   assign c2a2[56] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 56];
   assign c2a2[57] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 57];
   assign c2a2[58] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 58];
   assign c2a2[59] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 59];
   assign c2a2[60] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 60];
   assign c2a2[61] = cmd_block_table2[add_dispatch_pointer[1:3]*62 + 61];

   assign c2b2[0:4] = 0;
   assign c2b2[5] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 5];
   assign c2b2[6] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 6];
   assign c2b2[7] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 7];
   assign c2b2[8] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 8];
   assign c2b2[9] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 9];
   assign c2b2[10] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 10];
   assign c2b2[11] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 11];
   assign c2b2[12] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 12];
   assign c2b2[13] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 13];
   assign c2b2[14] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 14];
   assign c2b2[15] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 15];
   assign c2b2[16] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 16];
   assign c2b2[17] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 17];
   assign c2b2[18] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 18];
   assign c2b2[19] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 19];
   assign c2b2[20] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 20];
   assign c2b2[21] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 21];
   assign c2b2[22] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 22];
   assign c2b2[23] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 23];
   assign c2b2[24] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 24];
   assign c2b2[25] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 25];
   assign c2b2[26] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 26];
   assign c2b2[27] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 27];
   assign c2b2[28] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 28];
   assign c2b2[29] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 29];
   assign c2b2[30] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 30];
   assign c2b2[31] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 31];
   assign c2b2[32] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 32];
   assign c2b2[33] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 33];
   assign c2b2[34] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 34];
   assign c2b2[35] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 35];
   assign c2b2[36] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 36];
   assign c2b2[37] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 37];
   assign c2b2[38] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 38];
   assign c2b2[39] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 39];
   assign c2b2[40] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 40];
   assign c2b2[41] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 41];
   assign c2b2[42] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 42];
   assign c2b2[43] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 43];
   assign c2b2[44] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 44];
   assign c2b2[45] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 45];
   assign c2b2[46] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 46];
   assign c2b2[47] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 47];
   assign c2b2[48] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 48];
   assign c2b2[49] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 49];
   assign c2b2[50] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 50];
   assign c2b2[51] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 51];
   assign c2b2[52] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 52];
   assign c2b2[53] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 53];
   assign c2b2[54] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 54];
   assign c2b2[55] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 55];
   assign c2b2[56] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 56];
   assign c2b2[57] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 57];
   assign c2b2[58] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 58];
   assign c2b2[59] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 59];
   assign c2b2[60] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 60];
   assign c2b2[61] = cmd_block_table2[shift_dispatch_pointer[1:3]*62 + 61];
   
   //set the outputs that drive the adder
   assign prio_adder_out_vld = (dispatch_add != 5'b0) ? 1'b1 : 1'b0;
   assign prio_adder_tag = add_dispatch_pointer;

   assign prio_adder_follow_branch =
	  (~add_dispatch_pointer[0]) ?  c1a2[5:9]    :
	  (add_dispatch_pointer[0]) ? c2a2[5:9] :
	  5'b0;

   assign prio_adder_cmd =
	  (~add_dispatch_pointer[0]) ? c1a2[10:13] :
	  (add_dispatch_pointer[0]) ? c2a2[10:13] :
	  4'b0;

   assign prio_adder_data1 =
	  (( dispatch_add != 5'b0) && ~add_dispatch_pointer[0]) ? c1a2[14:18] :
	  (( dispatch_add != 5'b0) && add_dispatch_pointer[0]) ? c2a2[14:18] :
	  5'b0;

   assign prio_adder_data2 =
	  (( dispatch_add != 5'b0) && ~add_dispatch_pointer[0]) ? c1a2[19:23] :
	  (( dispatch_add != 5'b0) && add_dispatch_pointer[0]) ? c2a2[19:23] :
	  5'b0;
   
   assign prio_adder_result =
	  (( dispatch_add != 5'b0) && ~add_dispatch_pointer[0]) ? c1a2[24:28] :
	  (( dispatch_add != 5'b0) && add_dispatch_pointer[0]) ? c2a2[24:28] :
	  5'b0;
   
   //set the outputs that drive the shifter
   assign prio_shift_out_vld = (dispatch_shift != 5'b0) ? 1'b1 : 1'b0;
   assign prio_shift_tag = shift_dispatch_pointer;
   assign prio_shift_follow_branch = 
	  ( ~shift_dispatch_pointer[0] ) ? c1b2[5:9] :
	  ( shift_dispatch_pointer[0] ) ? c2b2[5:9] :
	  5'b0;

   assign prio_shift_cmd =
	  ( ~shift_dispatch_pointer[0]) ? c1b2[10:13] :
	  ( shift_dispatch_pointer[0]) ? c2b2[10:13] :
	  4'b0;

   assign prio_shift_data1 =
	  (( dispatch_shift != 5'b0) && ~shift_dispatch_pointer[0]) ? c1b2[14:18] :
	  (( dispatch_shift != 5'b0) && shift_dispatch_pointer[0]) ? c2b2[14:18] :
	  5'b0;

   assign prio_shift_data2 =
	  (( dispatch_shift != 5'b0) && ~shift_dispatch_pointer[0]) ? c1b2[19:23] :
	  (( dispatch_shift != 5'b0) && shift_dispatch_pointer[0]) ? c2b2[19:23] :
	  5'b0;
   
   assign prio_shift_result =
	  (( dispatch_shift != 5'b0) && ~shift_dispatch_pointer[0]) ? c1b2[24:28] :
	  (( dispatch_shift != 5'b0) && shift_dispatch_pointer[0]) ? c2b2[24:28] :
	  5'b0;

   assign prio_shift_data =
	  ( ~shift_dispatch_pointer[0] ) ? c1b2[29:60] :
	  ( shift_dispatch_pointer[0] ) ? c2b2[29:60] :
	  32'b0;
        
endmodule // priority
