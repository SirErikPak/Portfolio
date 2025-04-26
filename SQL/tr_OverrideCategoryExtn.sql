IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_OverrideCategoryExtn]'))
DROP TRIGGER [dbo].[tr_OverrideCategoryExtn]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_OverrideCategoryExtn] ON [dbo].[OverrideCategoryExtn] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info			Varchar(25),
			@SeqNo			Int,
			@Max			Int,
			@Min			Int,
			@iEmpId			Varchar(10),
			@iComment		Varchar(300),
			@iLastUser		Int,
			@iLastUpdate	Char(10),
			@cEmpId			Varchar(10),
			@cComment		Varchar(300),
			@cLastUser		Int,
			@cLastUpdate	Char(10)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(OC.EmpId) 
	From OverrideCategoryExtn OC INNER JOIN deleted DEL 
	ON OC.EmpId = DEL.EmpId) > 0

	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo			Int		Identity,
			EmpID			Varchar(10)	NOT NULL
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	EmpID
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(EmpID),
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
			Select	@cEmpId			= O.EmpId,
					@cComment		= O.Comment,
					@cLastUser		= O.LastUser,
					@cLastUpdate	= CONVERT(Char(10),O.LastUpdate,101)
			From	OverrideCategoryExtn O INNER JOIN deleted D
					ON O.EmpId = D.EmpId
			Where	O.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
			
			/*********************************************************/
			/*                                                       */
			/*********************************************************/	
			Select	@iEmpId			= D.EmpId,
					@iComment		= D.Comment,
					@iLastUser		= D.LastUser,
					@iLastUpdate	= CONVERT(Char(10),D.LastUpdate,101)
			From	OverrideCategoryExtn O INNER JOIN deleted D
					ON O.EmpId = D.EmpId
			Where	D.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
			
			/*********************************************************/
			/*                                                       */
			/*********************************************************/	
			IF	(
				-----(@cEmpId = @iEmpId) And 
				(@cComment = @iComment) And (@cLastUser = @iLastUser) -----And
				-----(@cLastUpdate = @iLastUpdate)	
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
				Insert	AuditOverrideCategoryExtn
				SELECT	D.EmpId,
						D.Comment,
						D.LastUser,
						D.LastUpdate,
						@Info,
						GETDATE(),
						USER_NAME()
				From	OverrideCategoryExtn O INNER JOIN deleted D
						ON O.EmpId = D.EmpId
				Where	D.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
							
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
		Insert	AuditOverrideCategoryExtn
		SELECT	EmpId,
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