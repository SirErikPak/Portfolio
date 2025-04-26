IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_ASLRequest]'))
DROP TRIGGER [dbo].[tr_ASLRequest]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TRIGGER [dbo].[tr_ASLRequest] ON [dbo].[ASLRequest] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info					Varchar(25),
			@SeqNo					Int,
			@Max					Int,
			@Min					Int,
			@iID					Int,					
			@iEmpId					Varchar(10),
			@iASLEffectiveDate		Char(10),
			@iCountryCode			Varchar(5),
			@iBuisnessUnit			Varchar(255),
			@iTierId				Int,
			@iJobCode				Varchar(20),
			@iRequestASLReasonId	Int,
			@iIsCapitalProjects		Bit,
			@iIsExceptionASL		Bit,
			@iMgrId					Varchar(10),
			@iMgrApprovalDate		Char(10),
			@iMgrComments			Varchar(300),
			@iValidStartDate		Char(10),
			@iValidEndDate			Char(10),
			@iWorkFlowStatusID		Int,
			@iASLRequestStatus		Varchar(25),
			@iComment				Varchar(300),
			@cID					Int,					
			@cEmpId					Varchar(10),
			@cASLEffectiveDate		Char(10),
			@cCountryCode			Varchar(5),
			@cBuisnessUnit			Varchar(255),
			@cTierId				Int,
			@cJobCode				Varchar(20),
			@cRequestASLReasonId	Int,
			@cIsCapitalProjects		Bit,
			@cIsExceptionASL		Bit,
			@cMgrId					Varchar(10),
			@cMgrApprovalDate		Char(10),
			@cMgrComments			Varchar(300),
			@cValidStartDate		Char(10),
			@cValidEndDate			Char(10),
			@cWorkFlowStatusID		Int,
			@cASLRequestStatus		Varchar(25),
			@cComment				Varchar(300)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(AU.EmpID) 
	From ASLRequest AU INNER JOIN deleted DEL 
	ON AU.ID = DEL.ID 
	AND AU.EmpId = DEL.EmpID) > 0

	Begin
	/*****************************************************************/
	/*                                                               */
	/*****************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo		Int		Identity,
			ID			Int		NOT NULL,
			EmpID		Varchar(10) NOT NULL	
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	ID,
				EmpId
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(ID),
				@SeqNo	= 1,
				@Min	= 1
		From	#Temp
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		WHILE @SeqNo <= @Max
			 Begin		-- BEGIN WHILE LOOP		
		
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Select	@iID					= D.ID,
					@iEmpId					= D.EmpId,
					@iASLEffectiveDate		= CONVERT(Char(10),ISNULL(D.ASLEffectiveDate,''),101),
					@iCountryCode			= D.CountryCode,
					@iBuisnessUnit			= ISNULL(D.BuisnessUnit,''),
					@iTierId				= D.TierId,
					@iJobCode				= D.JobCode,
					@iRequestASLReasonId	= D.RequestASLReasonId,
					@iIsCapitalProjects		= D.IsCapitalProjects,
					@iIsExceptionASL		= D.IsExceptionASL,
					@iMgrId					= D.MgrId,
					@iMgrApprovalDate		= CONVERT(Char(10),ISNULL(D.MgrApprovalDate,''),101),
					@iMgrComments			= ISNULL(D.MgrComments,''),
					@iValidStartDate		= CONVERT(Char(10),ISNULL(D.ValidStartDate,''),101),
					@iValidEndDate			= CONVERT(Char(10),ISNULL(D.ValidEndDate,''),101),
					@iWorkFlowStatusID		= ISNULL(D.WorkFlowStatusID,''),
					@iASLRequestStatus		= D.ASLRequestStatus,
					@iComment				= D.Comment
			From	ASLRequest A INNER JOIN deleted D 
					ON A.ID = D.ID 
					AND A.EmpId = D.EmpID
			Where	D.ID IN (Select ID From #Temp Where SeqNo = @SeqNo)
			And		D.EmpID IN (Select EmpID From #Temp Where SeqNo = @SeqNo)
	
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Select	@cID					= A.ID,
					@cEmpId					= A.EmpId,
					@cASLEffectiveDate		= CONVERT(Char(10),ISNULL(A.ASLEffectiveDate,''),101),
					@cCountryCode			= A.CountryCode,
					@cBuisnessUnit			= ISNULL(A.BuisnessUnit,''),
					@cTierId				= A.TierId,
					@cJobCode				= A.JobCode,
					@cRequestASLReasonId	= A.RequestASLReasonId,
					@cIsCapitalProjects		= A.IsCapitalProjects,
					@cIsExceptionASL		= A.IsExceptionASL,
					@cMgrId					= A.MgrId,
					@cMgrApprovalDate		= CONVERT(Char(10),ISNULL(A.MgrApprovalDate,''),101),
					@cMgrComments			= ISNULL(A.MgrComments,''),
					@cValidStartDate		= CONVERT(Char(10),ISNULL(A.ValidStartDate,''),101),
					@cValidEndDate			= CONVERT(Char(10),ISNULL(A.ValidEndDate,''),101),
					@cWorkFlowStatusID		= ISNULL(A.WorkFlowStatusID,''),
					@cASLRequestStatus		= A.ASLRequestStatus,
					@cComment				= A.Comment
			From	ASLRequest A INNER JOIN deleted D 
					ON A.ID = D.ID 
					AND A.EmpId = D.EmpID
			Where	A.ID IN (Select ID From #Temp Where SeqNo = @SeqNo)
			And		A.EmpID IN (Select EmpID From #Temp Where SeqNo = @SeqNo)
				
			/*********************************************************/
			/*                                                       */
			/*********************************************************/		
			IF  (
				(@cID = @iID) And (@cEmpId	= @iEmpId) And (@cASLEffectiveDate = @iASLEffectiveDate) And
				(@cCountryCode = @iCountryCode) And (@cBuisnessUnit = @iBuisnessUnit) And
				(@cTierId = @iTierId) And (@cJobCode = @iJobCode) And (@cRequestASLReasonId = @iRequestASLReasonId) And
				(@cIsCapitalProjects = @iIsCapitalProjects) And (@cIsExceptionASL = @iIsExceptionASL) And
				(@cMgrId = @iMgrId) And (@cMgrApprovalDate = @iMgrApprovalDate) And (@cMgrComments = @iMgrComments) And
				(@cValidStartDate = @iValidStartDate) And (@cValidEndDate = @iValidEndDate) And (@cWorkFlowStatusID = @iWorkFlowStatusID) And
				(@cASLRequestStatus = @iASLRequestStatus) And (@cComment = @iComment)
				)		
			 Begin
				/*****************************************************/
				/*					No Change(s)                     */
				/*****************************************************/
				GOTO NextRecordCheck	 
			 End
			Else
			 Begin
				/*****************************************************/
				/*				Update Change(s)                     */
				/*****************************************************/
					Insert	AuditASLRequest
					Select	D.ID,
							D.EmpId,
							D.ASLEffectiveDate,
							D.CountryCode,
							D.BuisnessUnit,
							D.TierId,
							D.JobCode,
							D.RequestASLReasonId,
							D.IsCapitalProjects,
							D.IsExceptionASL,
							D.MgrId,
							D.MgrApprovalDate,
							D.MgrComments,
							D.ValidStartDate,
							D.ValidEndDate,
							D.WorkFlowStatusID,
							D.ASLRequestStatus,
							D.Comment,
							D.LastUser,
							D.LastUpdate,
							@Info,
							GETDATE(),
							USER_NAME()
					From	ASLRequest A INNER JOIN deleted D 
							ON A.ID = D.ID 
							AND A.EmpId = D.EmpID
					Where	D.ID IN (Select ID From #Temp Where SeqNo = @SeqNo)
					And		D.EmpID IN (Select EmpID From #Temp Where SeqNo = @SeqNo)
										
			 End
				/*****************************************************/
				/*                                                   */
				/*****************************************************/
				NextRecordCheck:	
				Select	@SeqNo = @SeqNo + @Min

			 End		-- END WHILE LOOP
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Drop Table #Temp			 		
				
	End
 Else
	Begin	
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Delete'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	AuditASLRequest
		SELECT	ID,
				EmpId,
				ASLEffectiveDate,
				CountryCode,
				BuisnessUnit,
				TierId,
				JobCode,
				RequestASLReasonId,
				IsCapitalProjects,
				IsExceptionASL,
				MgrId,
				MgrApprovalDate,
				MgrComments,
				ValidStartDate,
				ValidEndDate,
				WorkFlowStatusID,
				ASLRequestStatus,
				Comment,
				LastUser,
				LastUpdate,
				@Info,
				GETDATE(),
				USER_NAME()
		From	Deleted
		
	End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO


