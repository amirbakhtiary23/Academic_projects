// Library:  calc3
// Module:  Adder Output Stage
// Author:  Naseer Siddique

module adder_output_stage(add_shift_branch_data, adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4, adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2, adder_out_tag3, adder_out_tag4, adder_write_adr, adder_write_data, adder_write_valid, scan_out, a_clk, adder_follow_branch, adder_out_cmd, adder_overflow, adder_result, adder_result_reg, adder_tag, b_clk, c_clk, reset, scan_in);

   output [0:15] add_shift_branch_data;

   output [0:31] adder_out_data1, adder_out_data2, adder_out_data3, adder_out_data4, adder_write_data;

   output [0:1]  adder_out_resp1, adder_out_resp2, adder_out_resp3, adder_out_resp4, adder_out_tag1, adder_out_tag2,  adder_out_tag3, adder_out_tag4;

   output [0:3]  adder_write_adr;

   output 	 adder_write_valid, scan_out;

   input 	 a_clk, adder_overflow, b_clk, c_clk, reset, scan_in;

   input [0:4] 	 adder_follow_branch, adder_result_reg;

   input [0:3] 	 adder_out_cmd, adder_tag;

   input [0:63]  adder_result;

   wire 	 valid_cmd, is_branch, not_branch, branch_true, skip_cmd, no_overflow;
   reg [0:1] 	 hold_resp;
   reg [0:3] 	 hold_tag;
   reg [0:4] 	 hold_result_reg;
   reg [0:31] 	 hold_reg_data, hold_out_data;
   reg [0:15] 	 branch_table;

   assign 	 valid_cmd = ( (adder_out_cmd[0:3] == 4'b0001) || (adder_out_cmd[0:3] == 4'b0010) || (adder_out_cmd[0:3] == 4'b1100) || (adder_out_cmd[0:3] == 4'b1101)) ? 1'b1 : 1'b0;

   assign 	 no_overflow = ( (adder_out_cmd[0:3] == 4'b0001) && (adder_out_cmd[0:3] == 4'b0010) && (adder_result[31] == 1'b1)) ? 1'b1 : 1'b0;

   assign 	 is_branch = ( (adder_out_cmd[0:3] == 4'b1100) || (adder_out_cmd[0:3] == 4'b1101)) ? 1'b1: 1'b0;

   assign 	 not_branch = ( (adder_out_cmd[0:3] == 4'b0001) && (adder_out_cmd[0:3] == 4'b0010)) ? 1'b1 : 1'b0;

   assign 	 branch_true = ( (adder_result[32:63] == 32'b0) && (is_branch) && (~adder_follow_branch || (branch_table[adder_follow_branch[1:4]] == 0))) ? 1'b1 : 1'b0;

   assign 	 skip_cmd = ( adder_follow_branch[0] && (branch_table[adder_follow_branch[1:4]] )) ? 1'b1  : 1'b0;
   									   
   always
     @ (negedge c_clk) begin

	branch_table[0] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0)) ? 1'b0 :
			   branch_table[0];

	branch_table[1] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0001)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0001)) ? 1'b0 :
			   branch_table[1];

	branch_table[2] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0010)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0010)) ? 1'b0 :
			   branch_table[2];
	
	branch_table[3] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0011)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0011)) ? 1'b0 :
			   branch_table[3];

	branch_table[4] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0100)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0100)) ? 1'b0 :
			   branch_table[4];

	branch_table[5] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0101)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0101)) ? 1'b0 :
			   branch_table[5];

	branch_table[6] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0110)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0110)) ? 1'b0 :
			   branch_table[6];

	branch_table[7] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b0111)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b0111)) ? 1'b0 :
			   branch_table[7];

	branch_table[8] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1000)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1000)) ? 1'b0 :
			   branch_table[8];

	branch_table[9] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1001)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1001)) ? 1'b0 :
			   branch_table[9];

	branch_table[10] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1010)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1010)) ? 1'b0 :
			   branch_table[10];

	branch_table[11] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1011)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1011)) ? 1'b0 :
			   branch_table[11];

	branch_table[12] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1100)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1100)) ? 1'b0 :
			   branch_table[12];

	branch_table[13] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1101)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1101)) ? 1'b0 :
			   branch_table[13];

	branch_table[14] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1110)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1110)) ? 1'b0 :
			   branch_table[14];

	branch_table[15] <= (reset) ? 1'b0 :
			   (is_branch && branch_true && (adder_tag[0:3] == 4'b1111)) ? 1'b1:
			   (is_branch && ~branch_true && (adder_tag[0:3] == 4'b1111)) ? 1'b0 :
			   branch_table[15];

	hold_result_reg[0:4] <= (reset || ~valid_cmd || (skip_cmd && not_branch) ) ? 5'b0 :
				(adder_result[31] && no_overflow && ((adder_out_cmd[0:3] == 4'b0001) || (adder_out_cmd[0:3] == 4'b0010))) ? 5'b0 :
				adder_result_reg;

	hold_tag[0:3] <= (reset || ~valid_cmd) ? 4'b0 : adder_tag[0:3];
	hold_resp[0:1] <= (reset || ~valid_cmd) ? 2'b0 :
			  (skip_cmd) ? 2'b11 :
			  (adder_result[31] && ~is_branch) ? 2'b10 :
			  2'b01;

	hold_reg_data[0:31] <= (reset || ~valid_cmd || (skip_cmd && not_branch) || is_branch) ? 32'b0 :
			      adder_result[32:63];
	hold_out_data[0:31] <= (reset || ~valid_cmd || skip_cmd) ? 32'b0 :
			       (is_branch && branch_true) ? 32'b1 :
			       32'b0;
		
     end // always @ (negedge c_clk)

   assign add_shift_branch_data[0:15] = branch_table[0:15];
   assign adder_write_adr[0:3] = hold_result_reg[1:4];
   assign adder_write_data[0:31] = hold_reg_data[0:31];
   assign adder_write_valid = hold_result_reg[0];

   assign adder_out_resp1[0:1] = (hold_tag[0:1] == 2'b00) ? hold_resp[0:1] : 2'b00;
   assign adder_out_resp2[0:1] = (hold_tag[0:1] == 2'b01) ? hold_resp[0:1] : 2'b00;
   assign adder_out_resp3[0:1] = (hold_tag[0:1] == 2'b10) ? hold_resp[0:1] : 2'b00;
   assign adder_out_resp4[0:1] = (hold_tag[0:1] == 2'b11) ? hold_resp[0:1] : 2'b00;

   assign adder_out_data1[0:31] = (hold_tag[0:1] == 2'b00) ? hold_out_data[0:31] : 32'b0;
   assign adder_out_data2[0:31] = (hold_tag[0:1] == 2'b01) ? hold_out_data[0:31] : 32'b0;
   assign adder_out_data3[0:31] = (hold_tag[0:1] == 2'b10) ? hold_out_data[0:31] : 32'b0;
   assign adder_out_data4[0:31] = (hold_tag[0:1] == 2'b11) ? hold_out_data[0:31] : 32'b0;

   assign adder_out_tag1[0:1] = (hold_tag[0:1] == 2'b00) ? hold_tag[2:3] : 2'b00;
   assign adder_out_tag2[0:1] = (hold_tag[0:1] == 2'b01) ? hold_tag[2:3] : 2'b00;
   assign adder_out_tag3[0:1] = (hold_tag[0:1] == 2'b10) ? hold_tag[2:3] : 2'b00;
   assign adder_out_tag4[0:1] = (hold_tag[0:1] == 2'b11) ? hold_tag[2:3] : 2'b00;

endmodule // adder_output_stage
