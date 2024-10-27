// Library:  calc3
// Module:  Adder Input Stage
// Author:  Naseer Siddique

module adder_input_stage( adder_cmd, adder_follow_branch, adder_out_cmd, adder_read_adr1, adder_read_adr2, adder_read_valid1, adder_read_valid2, adder_result_reg, adder_tag, scan_out, a_clk, b_clk, c_clk, prio_adder_cmd, prio_adder_data1, prio_adder_data2, prio_adder_follow_branch, prio_adder_out_vld, prio_adder_result, prio_adder_tag, reset, scan_in);

   output [0:3] adder_cmd, adder_out_cmd, adder_read_adr1, adder_read_adr2, adder_tag;

   input [0:3] prio_adder_cmd, prio_adder_tag;

   output [0:4] adder_follow_branch, adder_result_reg;

   input [0:4] 	prio_adder_data1, prio_adder_data2, prio_adder_follow_branch, prio_adder_result;

   output 	adder_read_valid1, adder_read_valid2, scan_out;

   input 	a_clk, b_clk, c_clk, prio_adder_out_vld, reset, scan_in;

   reg 		valid;
   reg [0:3] 	cmd, out_cmd, tag;
   reg [0:4] 	d1, d2, follow_branch, result;

   always
     @ (negedge c_clk) begin

	// latch dispatched command into registers
	cmd <= (reset || ~prio_adder_out_vld) ? 
	       4'b0 :
	       (prio_adder_cmd == 4'b0001) ? 
	       4'b0001 :
	       ((prio_adder_cmd == 4'b0010)  || 
		(prio_adder_cmd == 4'b1100) || 
		(prio_adder_cmd == 4'b1101)) ? 
	       4'b0010 :
	       4'b0;

	out_cmd <= (reset || ~prio_adder_out_vld) ? 4'b0 :
		   ((prio_adder_cmd == 4'b0001) || 
		    (prio_adder_cmd == 4'b1100) || 
		    (prio_adder_cmd == 4'b0010) || 
		    (prio_adder_cmd == 4'b1101)) ? 
		   prio_adder_cmd :
		   4'b0;

	d1 <= (reset) ? 5'b0 : prio_adder_data1;
	d2 <= (reset) ? 5'b0 : prio_adder_data2;
	follow_branch <= (reset) ? 5'b0 : prio_adder_follow_branch;
	valid <= (reset) ? 1'b0 : prio_adder_out_vld;
	result <= (reset) ? 5'b0 : prio_adder_result;
	tag <= (reset) ? 4'b0 : prio_adder_tag;

     end // always @ (negedge c_clk)

   assign adder_follow_branch = follow_branch[0:4];
   assign adder_read_adr1 = d1[1:4];
   assign adder_read_adr2 = d2[1:4];
   assign adder_read_valid1 = d1[0];
   assign adder_read_valid2 = d2[0];
   assign adder_result_reg = result[0:4];
   assign adder_tag = tag[0:3];
   assign adder_cmd = cmd[0:3];
   assign adder_out_cmd = out_cmd[0:3];
   

endmodule // adder_input_stage
