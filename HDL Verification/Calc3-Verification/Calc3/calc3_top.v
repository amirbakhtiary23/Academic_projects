// Library:  calc3
// Module:  Top-Level Schematic
// Author:  Naseer Siddique

`include "adder.v" 
`include "adder_input_stage.v" 
`include "adder_output_stage.v" 
`include "holdreg.v" 
`include "mux_out.v" 
`include "priority.v" 
`include "registers.v" 
`include "shift_input_stage.v" 
`include "shift_output_stage.v" 
`include "shifter.v" 

module calc3_top ( out1_data, out1_resp, out1_tag, out2_data, out2_resp, out2_tag, out3_data, out3_resp, out3_tag, out4_data, out4_resp, out4_tag, scan_out, a_clk, b_clk, c_clk, req1_cmd, req1_d1, req1_d2, req1_data, req1_r1, req1_tag, req2_cmd, req2_d1, req2_d2, req2_data, req2_r1, req2_tag, req3_cmd, req3_d1, req3_d2, req3_data, req3_r1, req3_tag, req4_cmd, req4_d1, req4_d2, req4_data, req4_r1, req4_tag, reset, scan_in);

   output [0:1] out1_resp, out1_tag, out2_resp, out2_tag, out3_resp, out3_tag, out4_resp, out4_tag;
   output [0:31] out1_data, out2_data, out3_data, out4_data;
   output 	 scan_out;
   
   input [0:1] req1_tag, req2_tag, req3_tag, req4_tag;
   input       a_clk, b_clk, c_clk, reset, scan_in;
   input [0:3] req1_cmd, req1_d1, req1_d2, req1_r1, req2_cmd, req2_d1, req2_d2, req2_r1, req3_cmd, req3_d1, req3_d2, req3_r1, req4_cmd, req4_d1, req4_d2, req4_r1;
   input [0:31] req1_data, req2_data, req3_data, req4_data;
      
   wire [0:15] add_shift_branch_data;
   wire [0:3]  adder_cmd, adder_out_cmd;
   wire [0:4]  adder_follow_branch;
   wire [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4;
   wire [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4;
   wire        adder_overflow;
   wire [0:3]  adder_read_adr1, adder_read_adr2; 
   wire [0:63] adder_read_d1, adder_read_d2; 
   wire        adder_read_valid1, adder_read_valid2;
   wire [0:63] adder_result;
   wire [0:4]  adder_result_reg;
   wire [0:3]  adder_tag, adder_write_adr;
   wire [0:31] adder_write_data;
   wire        adder_write_valid;
   wire [0:3]  hold1_cmd, hold1_d1, hold1_d2;
   wire [0:31] hold1_data;
   wire [0:3]  hold1_r1;
   wire [0:1]  hold1_tag;
   wire [0:3]  hold2_cmd, hold2_d1, hold2_d2;
   wire [0:31] hold2_data;
   wire [0:3]  hold2_r1;
   wire [0:1]  hold2_tag;
   wire [0:3]  hold3_cmd, hold3_d1, hold3_d2;
   wire [0:31] hold3_data;
   wire [0:3]  hold3_r1;
   wire [0:1]  hold3_tag;
   wire [0:3]  hold4_cmd;
   wire [0:3]  hold4_d1,  hold4_d2;
   wire [0:31] hold4_data;
   wire [0:3]  hold4_r1;
   wire [0:1]  hold4_tag;
   wire port1_invalid_op;
   wire [0:1] port1_invalid_tag;
   wire       port2_invalid_op;
   wire [0:1] port2_invalid_tag;
   wire       port3_invalid_op;
   wire [0:1] port3_invalid_tag;
   wire       port4_invalid_op;
   wire [0:1] port4_invalid_tag;
   wire [0:3] prio_adder_cmd;
   wire [0:4] prio_adder_data1, prio_adder_data2;
   wire [0:4] prio_adder_follow_branch;
   wire       prio_adder_out_vld;
   wire [0:4] prio_adder_result;
   wire [0:3] prio_adder_tag,  prio_shift_cmd;
   wire [0:31] prio_shift_data;
   wire [0:4]  prio_shift_data1,  prio_shift_data2;
   wire [0:4]  prio_shift_follow_branch;
   wire        prio_shift_out_vld;
   wire [0:4]  prio_shift_result;
   wire [0:3]  prio_shift_tag;
   wire        scan_connect1, scan_connect10, scan_connect11, scan_connect12, scan_connect2, scan_connect3, scan_connect4, scan_connect5, scan_connect6, scan_connect7, scan_connect8, scan_connect9;
   wire [0:3]  shift_cmd;
   wire [0:4]  shift_follow_branch;
   wire [0:3]  shift_out_cmd;
   wire [0:31] shift_out_data1, shift_out_data2, shift_out_data3, shift_out_data4;
   wire [0:1]  shift_out_resp1;
   wire [0:1]  shift_out_resp2, shift_out_resp3, shift_out_resp4, shift_out_tag1, shift_out_tag2, shift_out_tag3, shift_out_tag4;
   wire [0:3]  shift_read_adr1,  shift_read_adr2;
   wire [0:63] shift_read_d1, shift_read_d2;
   wire        shift_read_valid1,  shift_read_valid2;
   wire [0:63] shift_result;
   wire [0:4]  shift_result_reg;
   wire [0:3]  shift_tag, shift_write_adr;
   wire [0:31] shift_write_data;
   wire        shift_write_valid, store_data_valid;
   wire [0:63] store_val;


      holdreg holdreg1 (
		     .a_clk(a_clk),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .hold_d1(hold1_d1),
		     .hold_d2(hold1_d2),
		     .hold_data(hold1_data),
		     .hold_prio_req(hold1_cmd),
		     .hold_prio_tag(hold1_tag),
		     .hold_r1(hold1_r1),
		     .req_cmd_in(req1_cmd),
		     .req_d1(req1_d1),
		     .req_d2(req1_d2),
		     .req_data(req1_data),
		     .req_r1(req1_r1),
		     .req_tag(req1_tag),
		     .reset(reset),
		     .scan_in(scan_in),
		     .scan_out(scan_connect1)
		     );

   holdreg holdreg2 (
		     .a_clk(a_clk),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .hold_d1(hold2_d1),
		     .hold_d2(hold2_d2),
		     .hold_data(hold2_data),
		     .hold_prio_req(hold2_cmd),
		     .hold_prio_tag(hold2_tag),
		     .hold_r1(hold2_r1),
		     .req_cmd_in(req2_cmd),
		     .req_d1(req2_d1),
		     .req_d2(req2_d2),
		     .req_data(req2_data),
		     .req_r1(req2_r1),
		     .req_tag(req2_tag),
		     .reset(reset),
		     .scan_in(scan_connect1),
		     .scan_out(scan_conect2)
		     );

  holdreg holdreg3 (
		     .a_clk(a_clk),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .hold_d1(hold3_d1),
		     .hold_d2(hold3_d2),
		     .hold_data(hold3_data),
		     .hold_prio_req(hold3_cmd),
		     .hold_prio_tag(hold3_tag),
		     .hold_r1(hold3_r1),
		     .req_cmd_in(req3_cmd),
		     .req_d1(req3_d1),
		     .req_d2(req3_d2),
		     .req_data(req3_data),
		     .req_r1(req3_r1),
		     .req_tag(req3_tag),
		     .reset(reset),
		     .scan_in(scan_connect2),
		     .scan_out(scan_connect3)
		     );

   holdreg holdreg4 (
		     .a_clk(a_clk),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .hold_d1(hold4_d1),
		     .hold_d2(hold4_d2),
		     .hold_data(hold4_data),
		     .hold_prio_req(hold4_cmd),
		     .hold_prio_tag(hold4_tag),
		     .hold_r1(hold4_r1),
		     .req_cmd_in(req4_cmd),
		     .req_d1(req4_d1),
		     .req_d2(req4_d2),
		     .req_data(req4_data),
		     .req_r1(req4_r1),
		     .req_tag(req4_tag),
		     .reset(reset),
		     .scan_in(scan_connect3),
		     .scan_out(scan_connect4)
		     );
   
     
      priority priority1 (
		       .a_clk(a_clk),
		       .b_clk(b_clk),
		       .c_clk(c_clk),
		       .hold1_cmd(hold1_cmd),
		       .hold1_data1(hold1_d1),
		       .hold1_data2(hold1_d2),
		       .hold1_data(hold1_data),
		       .hold1_result(hold1_r1),
		       .hold1_tag(hold1_tag),
		       .hold2_cmd(hold2_cmd),
		       .hold2_data1(hold2_d1),
		       .hold2_data2(hold2_d2),
		       .hold2_data(hold2_data),
		       .hold2_result(hold2_r1),
		       .hold2_tag(hold2_tag),
		       .hold3_cmd(hold3_cmd),
		       .hold3_data1(hold3_d1),
		       .hold3_data2(hold3_d2),
		       .hold3_data(hold3_data),
		       .hold3_result(hold3_r1),
		       .hold3_tag(hold3_tag),
		       .hold4_cmd(hold4_cmd),
		       .hold4_data1(hold4_d1),
		       .hold4_data2(hold4_d2),
		       .hold4_data(hold4_data),
		       .hold4_result(hold4_r1),
		       .hold4_tag(hold4_tag),
		       .port1_invalid_op(port1_invalid_op),
		       .port1_invalid_tag(port1_invalid_tag),
		       .port2_invalid_op(port2_invalid_op),
		       .port2_invalid_tag(port2_invalid_tag),
		       .port3_invalid_op(port3_invalid_op),
		       .port3_invalid_tag(port3_invalid_tag),
		       .port4_invalid_op(port4_invalid_op),
		       .port4_invalid_tag(port4_invalid_tag),
		       .prio_adder_cmd(prio_adder_cmd),
		       .prio_adder_data1(prio_adder_data1),
		       .prio_adder_data2(prio_adder_data2),
		       .prio_adder_follow_branch(prio_adder_follow_branch),
		       .prio_adder_out_vld(prio_adder_out_vld),
		       .prio_adder_result(prio_adder_result),
		       .prio_adder_tag(prio_adder_tag),
		       .prio_shift_cmd(prio_shift_cmd),
		       .prio_shift_data1(prio_shift_data1),
		       .prio_shift_data2(prio_shift_data2),
		       .prio_shift_data(prio_shift_data),
		       .prio_shift_follow_branch(prio_shift_follow_branch),
		       .prio_shift_out_vld(prio_shift_out_vld),
		       .prio_shift_result(prio_shift_result),
		       .prio_shift_tag(prio_shift_tag),
		       .reset(reset),
		       .scan_in(scan_connect4),
		       .scan_out(scan_connect5)
		       );

  adder_input_stage adder_input_stage1(
					.a_clk(a_clk), 
					.adder_cmd(adder_cmd), 
					.adder_follow_branch(adder_follow_branch),
					.adder_out_cmd(adder_out_cmd),
					.adder_read_adr1(adder_read_adr1),
					.adder_read_adr2(adder_read_adr2),
					.adder_read_valid1(adder_read_valid1),
					.adder_read_valid2(adder_read_valid2),
					.adder_result_reg(adder_result_reg),
					.adder_tag(adder_tag),
					.b_clk(b_clk),
					.c_clk(c_clk),
					.prio_adder_cmd(prio_adder_cmd),
					.prio_adder_data1(prio_adder_data1),
					.prio_adder_data2(prio_adder_data2),
					.prio_adder_follow_branch(prio_adder_follow_branch),
					.prio_adder_out_vld(prio_adder_out_vld),
					.prio_adder_result(prio_adder_result),
					.prio_adder_tag(prio_adder_tag),
					.reset(reset),
					.scan_in(scan_connect5),
					.scan_out(scan_connect6)
					);

      shift_input_stage shift_input_stage1(
					.a_clk(a_clk),
					.b_clk(b_clk),
					.c_clk(c_clk),
					.prio_shift_cmd(prio_shift_cmd),
					.prio_shift_data1(prio_shift_data1),
					.prio_shift_data2(prio_shift_data2),
					.prio_shift_data(prio_shift_data),
					.prio_shift_follow_branch(prio_shift_follow_branch),
					.prio_shift_out_vld(prio_shift_out_vld),
					.prio_shift_result(prio_shift_result),
					.prio_shift_tag(prio_shift_tag),
					.reset(reset),
					.scan_in(scan_connect6),
					.scan_out(scan_connect7),
					.shift_cmd(shift_cmd),
					.shift_follow_branch(shift_follow_branch),
					.shift_out_cmd(shift_out_cmd),
					.shift_read_adr1(shift_read_adr1),
					.shift_read_adr2(shift_read_adr2),
					.shift_read_valid1(shift_read_valid1),
					.shift_read_valid2(shift_read_valid2),
					.shift_result_reg(shift_result_reg),
					.shift_tag(shift_tag),
					.store_data_valid(store_data_valid),
					.store_val(store_val)
					);

  registers registers1 (
			 .a_clk(a_clk),
			 .b_clk(b_clk),
			 .c_clk(c_clk),
			 .adder_read_adr1(adder_read_adr1),
			 .adder_read_adr2(adder_read_adr2),
			 .adder_read_d1(adder_read_d1),
			 .adder_read_d2(adder_read_d2),
			 .adder_read_valid1(adder_read_valid1),
			 .adder_read_valid2(adder_read_valid2),
			 .adder_write_adr(adder_write_adr),
			 .adder_write_data(adder_write_data),
			 .adder_write_valid(adder_write_valid),
			 .reset(reset),
			 .shift_read_adr1(shift_read_adr1),
			 .shift_read_adr2(shift_read_adr2),
			 .shift_read_d1(shift_read_d1),
			 .shift_read_d2(shift_read_d2),
			 .shift_read_valid1(shift_read_valid1),
			 .shift_read_valid2(shift_read_valid2),
			 .shift_write_adr(shift_write_adr),
			 .shift_write_data(shift_write_data),
			 .shift_write_valid(shift_write_valid)
			 );

   adder adder1(
		.alu_cmd(adder_cmd), 
		.bin_ovfl(adder_overflow), 
		.bin_sum(adder_result), 
		.fxu_areg_q(adder_read_d1), 
		.fxu_breg_q(adder_read_d2)
		);

 
   shifter shifter1 (
		     .shift_cmd(shift_cmd),
		     .shift_out(shift_result),
		     .shift_places(shift_read_d2),
		     .shift_val(shift_read_d1),
		     .store_data_valid(store_data_valid),
		     .store_val(store_val)
		     );

   
   adder_output_stage adder_output_stage1(
					  .a_clk(a_clk),
					  .add_shift_branch_data(add_shift_branch_data),
					  .adder_follow_branch(adder_follow_branch),
					  .adder_out_cmd(adder_out_cmd),
					  .adder_out_data1(adder_out_data1),
					  .adder_out_data2(adder_out_data2),
					  .adder_out_data3(adder_out_data3),
					  .adder_out_data4(adder_out_data4),
					  .adder_out_resp1(adder_out_resp1),
					  .adder_out_resp2(adder_out_resp2),
					  .adder_out_resp3(adder_out_resp3),
					  .adder_out_resp4(adder_out_resp4),
					  .adder_out_tag1(adder_out_tag1),
					  .adder_out_tag2(adder_out_tag2),
					  .adder_out_tag3(adder_out_tag3),
					  .adder_out_tag4(adder_out_tag4),
					  .adder_overflow(adder_overflow),
					  .adder_result(adder_result),
					  .adder_result_reg(adder_result_reg),
					  .adder_tag(adder_tag),
					  .adder_write_adr(adder_write_adr),
					  .adder_write_data(adder_write_data),
					  .adder_write_valid(adder_write_valid),
					  .b_clk(b_clk),
					  .c_clk(c_clk),
					  .reset(reset),
					  .scan_in(scan_connect7),
					  .scan_out(scan_connect8)
					  );

   shift_output_stage shift_output_stage1 (
					   .a_clk(a_clk),
					   .add_shift_branch_data(add_shift_branch_data),
					   .b_clk(b_clk),
					   .c_clk(c_clk),
					   .reset(reset),
					   .scan_in(scan_connect8),
					   .scan_out(scan_connect9),
					   .shift_follow_branch(shift_follow_branch),
					   .shift_out_cmd(shift_out_cmd),
					   .shift_out_data1(shift_out_data1),
					   .shift_out_data2(shift_out_data2),
					   .shift_out_data3(shift_out_data3),
					   .shift_out_data4(shift_out_data4),
					   .shift_out_resp1(shift_out_resp1),
					   .shift_out_resp2(shift_out_resp2),
					   .shift_out_resp3(shift_out_resp3),
					   .shift_out_resp4(shift_out_resp4),
					   .shift_out_tag1(shift_out_tag1),
					   .shift_out_tag2(shift_out_tag2),
					   .shift_out_tag3(shift_out_tag3),
					   .shift_out_tag4(shift_out_tag4),
					   .shift_result(shift_result),
					   .shift_result_reg(shift_result_reg),
					   .shift_tag(shift_tag),
					   .shift_write_adr(shift_write_adr),
					   .shift_write_data(shift_write_data),
					   .shift_write_valid(shift_write_valid)
					   );

   
    mux_out mux_out1 (
		     .a_clk(a_clk),
		     .adder_data(adder_out_data1),
		     .adder_resp(adder_out_resp1),
		     .adder_tag(adder_out_tag1),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .invalid_op(port1_invalid_op),
		     .invalid_op_tag(port1_invalid_tag),
		     .out_data(out1_data),
		     .out_resp(out1_resp),
		     .out_tag(out1_tag),
		     .reset(reset),
		     .scan_in(scan_connect9),
		     .scan_out(scan_connect10),
		     .shift_data(shift_out_data1),
		     .shift_resp(shift_out_resp1),
		     .shift_tag(shift_out_tag1)
		     );

   mux_out mux_out2 (
		     .a_clk(a_clk),
		     .adder_data(adder_out_data2),
		     .adder_resp(adder_out_resp2),
		     .adder_tag(adder_out_tag2),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .invalid_op(port2_invalid_op),
		     .invalid_op_tag(port2_invalid_tag),
		     .out_data(out2_data),
		     .out_resp(out2_resp),
		     .out_tag(out2_tag),
		     .reset(reset),
		     .scan_in(scan_connect10),
		     .scan_out(scan_connect11),
		     .shift_data(shift_out_data2),
		     .shift_resp(shift_out_resp2),
		     .shift_tag(shift_out_tag2)
		     );

   mux_out mux_out3 (
		     .a_clk(a_clk),
		     .adder_data(adder_out_data3),
		     .adder_resp(adder_out_resp3),
		     .adder_tag(adder_out_tag3),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .invalid_op(port3_invalid_op),
		     .invalid_op_tag(port3_invalid_tag),
		     .out_data(out3_data),
		     .out_resp(out3_resp),
		     .out_tag(out3_tag),
		     .reset(reset),
		     .scan_in(scan_connect11),
		     .scan_out(scan_connect12),
		     .shift_data(shift_out_data3),
		     .shift_resp(shift_out_resp3),
		     .shift_tag(shift_out_tag3)
		     );

   mux_out mux_out4 (
		     .a_clk(a_clk),
		     .adder_data(adder_out_data4),
		     .adder_resp(adder_out_resp4),
		     .adder_tag(adder_out_tag4),
		     .b_clk(b_clk),
		     .c_clk(c_clk),
		     .invalid_op(port4_invalid_op),
		     .invalid_op_tag(port4_invalid_tag),
		     .out_data(out4_data),
		     .out_resp(out4_resp),
		     .out_tag(out4_tag),
		     .reset(reset),
		     .scan_in(scan_connect12),
		     .scan_out(scan_out),
		     .shift_data(shift_out_data4),
		     .shift_resp(shift_out_resp4),
		     .shift_tag(shift_out_tag4)
		     );


 


endmodule // calc3_top

     
					   
					   
			